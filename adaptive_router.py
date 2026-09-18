"""Answer-aware, context-conditioned finite acquisition with guarded updates.

This module does not call an LLM. It consumes normalized acquired responses.
Observation callbacks never receive labels. Learned tables are compiled once;
the deployment JSON contains data, not executable code or pickled estimators.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Callable, Mapping, Sequence

import numpy as np
import transfer_study as legacy

N = 4
BUDGET = 3
State = tuple[int, tuple[int, ...]]


@dataclass(frozen=True)
class Pool:
    """An ordered worker population. Revision changes invalidate cached policy."""
    models: tuple[str, ...]
    revision: str

    def __post_init__(self):
        if (not isinstance(self.models, tuple) or len(self.models) != N or len(set(self.models)) != N
                or not all(isinstance(x, str) and x for x in self.models)
                or not isinstance(self.revision, str) or not self.revision):
            raise ValueError('Expected four distinct models and an explicit revision')

    @property
    def fingerprint(self) -> str:
        value = json.dumps([list(self.models), self.revision], separators=(',', ':'))
        return hashlib.sha256(value.encode()).hexdigest()


def encode(values: Sequence[str | None], kind: str) -> tuple[int, ...]:
    if any(v is not None and not isinstance(v, str) for v in values):
        raise ValueError('Responses must be normalized strings or None')
    if kind == 'equality':
        return legacy.canonical(values)
    if kind == 'polarity':
        try:
            return tuple({None: -1, 'no': 0, 'yes': 1}[v] for v in values)
        except KeyError as exc:
            raise ValueError('Polarity requires normalized yes/no/None') from exc
    raise ValueError('Unknown observation encoding')


def project(pattern: tuple[int, ...], mask: int, kind: str) -> State:
    values = tuple(pattern[i] for i in range(N) if mask >> i & 1)
    if kind == 'polarity':
        return mask, values
    return mask, legacy.canonical([None if v < 0 else str(v) for v in values])


@lru_cache(maxsize=2)
def graph(kind: str):
    if kind not in ('equality', 'polarity'):
        raise ValueError('Unknown graph')
    patterns = legacy.PATTERNS if kind == 'equality' else tuple(itertools.product((-1, 0, 1), repeat=N))
    states = tuple(sorted({project(p, m, kind) for p in patterns for m in range(1 << N)},
                          key=lambda s: (s[0].bit_count(), s)))
    index = {s: i for i, s in enumerate(states)}
    match = np.array([[project(p, s[0], kind) == s for p in patterns] for s in states], dtype=float)
    children = {}
    actions = {}
    for state in states:
        mask = state[0]
        left = [i for i in range(N) if not mask >> i & 1]
        actions[state] = tuple(batch for k in range(1, min(BUDGET - mask.bit_count(), len(left)) + 1)
                               for batch in itertools.combinations(left, k))
        for batch in actions[state]:
            new_mask = mask | sum(1 << i for i in batch)
            children[state, batch] = tuple(sorted({project(p, new_mask, kind) for p in patterns
                                                  if project(p, mask, kind) == state}))
    return patterns, states, index, match, actions, children


@dataclass
class Joint:
    kind: str
    mass: np.ndarray
    error_sum: np.ndarray

    def validate(self):
        n = len(graph(self.kind)[0])
        if self.mass.shape != (n,) or self.error_sum.shape != (n, N):
            raise ValueError('Joint shape mismatch')
        if (not np.isfinite(self.mass).all() or not np.isfinite(self.error_sum).all()
                or np.any(self.mass <= 0) or np.any(self.error_sum < 0)
                or np.any(self.error_sum > self.mass[:, None] + 1e-10)):
            raise ValueError('Invalid probability/error mass')


def fit_joint(answers, errors, kind='equality', strength=10., prior: Joint | None = None) -> Joint:
    a, e = np.asarray(answers, dtype=object), np.asarray(errors, dtype=float)
    if (a.ndim != 2 or a.shape[1] != N or a.shape != e.shape or len(a) == 0
            or not np.isfinite(e).all() or np.any((e < 0) | (e > 1))
            or not math.isfinite(strength) or strength <= 0):
        raise ValueError('Invalid fit arrays or strength')
    patterns = graph(kind)[0]
    ids = {p: i for i, p in enumerate(patterns)}
    bins = np.array([ids[encode(row, kind)] for row in a])
    mass = np.bincount(bins, minlength=len(patterns)).astype(float)
    loss_sum = np.column_stack([np.bincount(bins, weights=e[:, i], minlength=len(patterns))
                                for i in range(N)])
    if prior is None:
        global_error = (e.sum(axis=0) + .5) / (len(e) + 1)
        mass += strength / len(patterns)
        loss_sum += strength / len(patterns) * global_error
    else:
        prior.validate()
        if prior.kind != kind:
            raise ValueError('Prior encoding mismatch')
        normalizer = float(prior.mass.sum())
        mass += strength * prior.mass / normalizer
        loss_sum += strength * prior.error_sum / normalizer
    joint = Joint(kind, mass, loss_sum)
    joint.validate()
    return joint


class CompiledPolicy:
    """O(1) next action lookup; model-pool contract checked before any query."""
    def __init__(self, pool: Pool, kind: str, records: Mapping[State, tuple],
                 costs=(.01,) * N, defer=.25, name='compiled', forbid_invalid=True):
        if type(forbid_invalid) is not bool or not isinstance(name, str) or not name:
            raise ValueError('Invalid policy identity/compatibility flag')
        costs = tuple(float(x) for x in costs)
        if (len(costs) != N or not np.isfinite(costs).all() or min(costs) < 0
                or not math.isfinite(defer) or not 0 <= defer <= 1):
            raise ValueError('Invalid loss/cost configuration')
        patterns, states, _, _, actions, _ = graph(kind)
        required = {s for s in states if s[0].bit_count() <= BUDGET}
        if set(records) != required:
            raise ValueError('Incomplete or unknown policy states')
        cleaned = {}
        for state, rec in records.items():
            if len(rec) != 3:
                raise ValueError('Malformed action record')
            batch, choice, risk = rec
            if (not isinstance(batch, (list, tuple)) or type(choice) is not int
                    or not isinstance(risk, (int, float)) or not math.isfinite(risk)
                    or not 0 <= risk <= 1 or any(type(i) is not int for i in batch)):
                raise ValueError('Malformed action types/risk')
            batch = tuple(batch)
            seen = [i for i in range(N) if state[0] >> i & 1]
            if batch:
                if batch not in actions[state] or choice != -1:
                    raise ValueError('Unauthorized acquisition')
            elif choice != -1:
                if choice not in seen or (forbid_invalid and state[1][seen.index(choice)] == -1):
                    raise ValueError('Unacquired/invalid terminal candidate')
            cleaned[state] = (batch, choice, float(risk))
        self.pool, self.kind, self._records = pool, kind, cleaned
        self.costs, self.defer, self.name, self.forbid_invalid = costs, float(defer), name, bool(forbid_invalid)

    def execute(self, acquire: Callable[[int], str | None], *, pool: Pool):
        if pool != self.pool:
            raise ValueError('Worker pool/revision changed: recalibrate before use')
        observed = [None] * N
        mask, state = 0, (0, ())
        order, batches = [], []
        for _ in range(BUDGET + 1):
            batch, chosen, risk = self._records[state]
            if not batch:
                return chosen, risk, tuple(order), tuple(batches), state
            for i in batch:
                value = acquire(i)
                if value is not None and not isinstance(value, str):
                    raise ValueError('Invalid response type')
                observed[i] = value
                mask |= 1 << i
                order.append(i)
            batches.append(batch)
            state = mask, encode([observed[i] for i in range(N) if mask >> i & 1], self.kind)
        raise RuntimeError('Policy did not terminate within its budget')

    @classmethod
    def from_legacy(cls, policy: legacy.Policy, pool: Pool):
        if policy.mode not in ('bellman', 'myopic', 'single', 'fixed3', 'static'):
            raise ValueError('Legacy compilation requires a precomputed finite policy')
        records = {}
        for state in legacy.STATES:
            if state[0].bit_count() > BUDGET:
                continue
            batch = policy.next_batch(state)
            risk, choice = policy.world.terminal(state, policy.defer)
            records[state] = (batch, -1 if batch else choice, risk)
        return cls(pool, 'equality', records, (policy.price,) * N, policy.defer,
                   'compiled_' + policy.mode, forbid_invalid=False)

    def to_json(self) -> str:
        states = [[s[0], list(s[1]), list(rec[0]), rec[1], round(rec[2], 12)]
                  for s, rec in sorted(self._records.items())]
        document = {'schema': 'dtmoa.compiled/1', 'pool': {'models': list(self.pool.models), 'revision': self.pool.revision},
                    'fingerprint': self.pool.fingerprint, 'encoding': self.kind, 'costs': list(self.costs),
                    'budget': BUDGET, 'defer': self.defer, 'name': self.name,
                    'forbid_invalid': self.forbid_invalid, 'states': states}
        return json.dumps(document, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n'

    @classmethod
    def from_json(cls, text: str, *, expected_pool: Pool):
        if not isinstance(text, str) or len(text) > 500_000:
            raise ValueError('Invalid policy document size')
        try:
            def object_pairs(pairs):
                out = {}
                for k, v in pairs:
                    if k in out: raise ValueError('Duplicate JSON key')
                    out[k] = v
                return out
            doc = json.loads(text, object_pairs_hook=object_pairs,
                             parse_constant=lambda _: (_ for _ in ()).throw(ValueError('Non-finite JSON')))
            fields = {'schema','pool','fingerprint','encoding','costs','budget','defer','name','forbid_invalid','states'}
            if set(doc) != fields or doc['schema'] != 'dtmoa.compiled/1' or doc['budget'] != BUDGET:
                raise ValueError('Unsupported policy contract')
            pool = Pool(tuple(doc['pool']['models']), doc['pool']['revision'])
            if pool != expected_pool or doc['fingerprint'] != pool.fingerprint:
                raise ValueError('Pool identity mismatch')
            records = {}
            for row in doc['states']:
                if len(row) != 5 or type(row[0]) is not int or any(type(v) is not int for v in row[1]):
                    raise ValueError('Malformed state')
                state = row[0], tuple(row[1])
                if state in records: raise ValueError('Duplicate state')
                records[state] = (row[2], row[3], row[4])
            return cls(pool, doc['encoding'], records, doc['costs'], doc['defer'],
                       doc['name'], doc['forbid_invalid'])
        except (KeyError, TypeError, IndexError, OverflowError) as exc:
            raise ValueError('Malformed policy document') from exc


def compile_joint(joint: Joint, pool: Pool, *, costs=(.01,) * N, defer=.25, name='answer_aware') -> CompiledPolicy:
    joint.validate()
    if len(costs) != N or not np.isfinite(costs).all() or min(costs) < 0:
        raise ValueError('Invalid costs')
    _, states, index, match, actions, children = graph(joint.kind)
    mass = match @ joint.mass
    risks = (match @ joint.error_sum) / mass[:, None]
    values, records = {}, {}
    for state in reversed(states):
        mask = state[0]
        if mask.bit_count() > BUDGET: continue
        seen = [i for i in range(N) if mask >> i & 1]
        valid = [i for pos, i in enumerate(seen) if state[1][pos] != -1]
        stop_risk, choice = min([(defer, -1)] + [(float(risks[index[state], i]), i) for i in valid])
        best, batch = stop_risk, ()
        for candidate in actions[state]:
            ch = children[state, candidate]
            weights = np.array([mass[index[s]] for s in ch])
            weights /= weights.sum()
            q = sum(costs[i] for i in candidate) + sum(p * values[s] for s, p in zip(ch, weights))
            if q < best - 1e-12:
                best, batch = float(q), candidate
        values[state] = best
        records[state] = (batch, -1 if batch else choice, stop_risk)
    return CompiledPolicy(pool, joint.kind, records, costs, defer, name, forbid_invalid=True)


@dataclass
class ContextRouter:
    """Representation-only prompt head, separately fitted risk tables, 2 bins."""
    head: legacy.PromptHead
    threshold: float
    leaves: tuple[CompiledPolicy, CompiledPolicy]
    calibration_counts: tuple[int, int]

    def bins(self, text):
        return (self.head.predict(text).mean(axis=1) > self.threshold).astype(int)

    def execute(self, prompt: str, acquire, *, pool: Pool):
        if pool != self.leaves[0].pool: raise ValueError('Pool changed')
        return self.leaves[int(self.bins([prompt])[0])].execute(acquire, pool=pool)


def fit_context(representation, calibration, pool_name: str, pool: Pool,
                kind='equality', shrinkage=32., min_support=32) -> ContextRouter:
    if set(representation.groups) & set(calibration.groups):
        raise ValueError('Representation/risk calibration overlap')
    head = legacy.PromptHead().fit(representation.text, representation.errors[pool_name])
    threshold = float(np.median(head.predict(representation.text).mean(axis=1)))
    bins = (head.predict(calibration.text).mean(axis=1) > threshold).astype(int)
    base = fit_joint(calibration.answers[pool_name], calibration.errors[pool_name], kind, 10.)
    policies, counts = [], []
    for leaf in range(2):
        selected = bins == leaf
        count = int(selected.sum()); counts.append(count)
        joint = (fit_joint(calibration.answers[pool_name][selected], calibration.errors[pool_name][selected],
                           kind, shrinkage, prior=base) if count >= min_support else base)
        policies.append(compile_joint(joint, pool, name=f'context_{kind}_{leaf}'))
    return ContextRouter(head, threshold, tuple(policies), tuple(counts))


def promotion_bound(differences, *, alpha=.05, max_total_loss=1.03):
    """One-candidate empirical Bernstein bound on paired loss, independent gate.

D in [-M,M], iid groups, var uses ddof=1. No distribution-free drift guarantee.
An identically acting policy is handled separately by exact compilation, not by
pretending zero sample variance certifies a learned policy.
"""
    values = np.asarray(differences, dtype=float)
    if (values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all()
            or not 0 < alpha < 1 or not math.isfinite(max_total_loss) or max_total_loss <= 0
            or np.max(np.abs(values)) > max_total_loss + 1e-12):
        raise ValueError('Invalid gate observations or bound')
    n = len(values); logterm = math.log(2 / alpha)
    mean, var = float(values.mean()), float(values.var(ddof=1))
    radius = math.sqrt(2 * var * logterm / n) + 7 * (2 * max_total_loss) * logterm / (3 * (n - 1))
    return {'n': n, 'mean_difference': mean, 'upper_difference': mean + radius,
            'radius': radius, 'alpha': alpha, 'promote': mean + radius < 0,
            'assumption': 'one tuning-selected candidate and incumbent; independent iid gate groups; stationary deployment'}


def context_to_json(router: ContextRouter) -> str:
    value = {'schema': 'dtmoa.context/1', 'threshold': router.threshold,
             'counts': list(router.calibration_counts),
             'idf': router.head.tfidf.idf_.tolist(), 'coef': router.head.reg.coef_.tolist(),
             'intercept': router.head.reg.intercept_.tolist(),
             'leaves': [json.loads(p.to_json()) for p in router.leaves]}
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n'


def context_from_json(text: str, *, expected_pool: Pool) -> ContextRouter:
    """Load finite numeric state only; never deserialize Python code/pickle."""
    from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer
    from sklearn.linear_model import Ridge
    if len(text) > 1_000_000: raise ValueError('Context artifact too large')
    try:
        value = json.loads(text)
        if set(value) != {'schema','threshold','counts','idf','coef','intercept','leaves'} or value['schema'] != 'dtmoa.context/1':
            raise ValueError('Unknown context schema')
        idf, coef, intercept = (np.array(value[k], dtype=float) for k in ('idf','coef','intercept'))
        if (idf.shape != (2048,) or coef.shape != (4,2048) or intercept.shape != (4,)
                or not np.isfinite(idf).all() or not np.isfinite(coef).all() or not np.isfinite(intercept).all()
                or np.any(idf <= 0) or not math.isfinite(value['threshold'])
                or not 0 <= value['threshold'] <= 1 or len(value['counts']) != 2
                or any(type(n) is not int or n < 0 for n in value['counts']) or len(value['leaves']) != 2):
            raise ValueError('Invalid numerical context artifact')
        head = legacy.PromptHead()
        head.hash = HashingVectorizer(n_features=2048, analyzer='char', ngram_range=(3,5), alternate_sign=False, norm=None)
        head.tfidf = TfidfTransformer(); head.tfidf.idf_ = idf; head.tfidf.n_features_in_ = 2048
        head.reg = Ridge(alpha=10., solver='lsqr', tol=1e-8)
        head.reg.coef_, head.reg.intercept_, head.reg.n_features_in_ = coef, intercept, 2048
        leaves = tuple(CompiledPolicy.from_json(json.dumps(p), expected_pool=expected_pool) for p in value['leaves'])
        return ContextRouter(head, value['threshold'], leaves, tuple(value['counts']))
    except (KeyError, TypeError, OverflowError, IndexError) as exc:
        raise ValueError('Malformed context artifact') from exc
