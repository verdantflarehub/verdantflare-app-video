"""Measured capacity profiles and aligned search, without a guessed video-length limit."""
import json
import hashlib
from pathlib import Path
import re

class CapacityError(RuntimeError):
    pass


def search_capacity(trial, ceiling, repeats=3):
    """Find the largest safe 4n+1 window within a caller-specified test budget.

    trial returns 'ok', 'oom' or 'headroom_exceeded'; all other failures abort.
    Only a failed next aligned size establishes a bound. A reached ceiling is a lower bound.
    """
    if type(ceiling) is not int or ceiling < 9 or repeats < 2:
        raise ValueError('ceiling >= 9 and at least two repeats required')
    cap = (ceiling - 1) // 4
    good, bad, largest_success, trials = 1, None, None, []

    def check(index):
        nonlocal largest_success
        statuses = []
        for _ in range(repeats):
            status = trial(index * 4 + 1)
            if status not in {'ok', 'oom', 'headroom_exceeded'}:
                raise CapacityError('non_capacity_trial_failure')
            statuses.append(status)
            if status != 'ok':
                break
        trials.append({'frames': index * 4 + 1, 'statuses': statuses})
        if all(s != 'oom' for s in statuses):
            largest_success = max(largest_success or 0, index * 4 + 1)
        return all(s == 'ok' for s in statuses)

    current = 2
    while True:
        if check(current):
            good = current
            if current == cap:
                break
            current = min(current * 2, cap)
        else:
            bad = current
            break
    if good == 1:
        raise CapacityError('minimum_window_does_not_fit')
    if bad is not None:
        while bad - good > 1:
            middle = (good + bad) // 2
            if check(middle):
                good = middle
            else:
                bad = middle
    return {'window_frames': good * 4 + 1, 'overlap_frames': 4,
            'next_rejected_frames': bad * 4 + 1 if bad else None,
            'search_ceiling_frames': cap * 4 + 1,
            'bound': 'measured_budget_boundary' if bad else 'lower_bound_only',
            'largest_successful_test_frames': largest_success, 'repeats': repeats, 'trials': trials}


class CapacityProfile:
    def __init__(self, path):
        try:
            raw = Path(path).read_bytes()
            self.data = json.loads(raw)
            self.digest = hashlib.sha256(raw).hexdigest()
            if self.data['schema_version'] != 1 or self.data['status'] != 'measured':
                raise ValueError('profile is not measured')
            if not re.fullmatch(r'[0-9a-f]{64}', self.data['source_sha256']):
                raise ValueError('source digest missing')
            for key in ('baseline_used_bytes', 'reserve_bytes'):
                if type(self.data[key]) is not int or self.data[key] < 0:
                    raise ValueError('invalid memory budget')
        except (OSError, KeyError, TypeError, ValueError) as exc:
            raise CapacityError('capacity_profile_missing_or_invalid') from exc

    def select(self, identity, media, req):
        try:
            if self.data['identity'] != identity:
                raise ValueError('hardware/model/runtime changed')
            geometry = dict(input_width=media['width'], input_height=media['height'],
                            target_width=req.target_width, target_height=req.target_height)
            if self.data['geometry'] != geometry:
                raise ValueError('resolution has not been calibrated')
            result = self.data['result']
            size = result['window_frames']
            if type(size) is not int or size < 9 or (size - 1) % 4 or result['overlap_frames'] != 4:
                raise ValueError('invalid window')
            successes = [t for t in result['trials'] if t['frames'] == size]
            if not successes or result['repeats'] < 2 or any(
                len(t['statuses']) != result['repeats'] or any(s != 'ok' for s in t['statuses']) for t in successes
            ):
                raise ValueError('selected window lacks repeated successful trials')
            return dict(window_frames=size, overlap_frames=4, capacity_profile_sha256=self.digest,
                        reserve_bytes=self.data['reserve_bytes'], seam_method='linear_overlap',
                        window_seed='source_offset', scene_cut_threshold=.30)
        except (KeyError, TypeError, ValueError) as exc:
            raise CapacityError('capacity_profile_mismatch') from exc
