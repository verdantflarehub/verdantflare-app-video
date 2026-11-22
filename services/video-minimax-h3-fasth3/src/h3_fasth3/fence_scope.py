"""Durable physical-card namespaces and conservative legacy fence migration."""

from __future__ import annotations

from copy import deepcopy


def gpu_scope(raw: object) -> str:
    # FastH3 is a single-card runtime. Never silently select one from a list.
    if not isinstance(raw, str) or not raw.strip() or "," in raw or raw.strip() == "*":
        raise ValueError("invalid_fence_gpu_scope")
    return raw.strip()


def positive_token(raw: object) -> int:
    if isinstance(raw, bool) or not isinstance(raw, int) or raw <= 0:
        raise ValueError("invalid_scoped_fencing_token")
    return raw


class ScopedFenceState:
    """Updates return independent snapshots; publication belongs to the fsync caller."""

    def __init__(self, cards=None, leases=None):
        self.cards = deepcopy(cards or {})
        self.leases = deepcopy(leases or {})

    def payload(self) -> dict:
        return {"version": 1, "cards": deepcopy(self.cards), "leases": deepcopy(self.leases)}

    def minimum(self, scope: str) -> int:
        return max(self.cards.get(scope, {}).get("minimum", 0), self.cards.get("*", {}).get("minimum", 0))

    def consumed(self, scope: str, token: int) -> bool:
        return token in self.cards.get(scope, {}).get("consumed_tokens", []) or token <= self.cards.get("*", {}).get("minimum", 0)

    def advance(self, lease: str, token: int, scope: str, *, seal: bool) -> ScopedFenceState:
        positive_token(token)
        if not lease or (scope != "*" and gpu_scope(scope) != scope):
            raise ValueError("invalid_scoped_lease")
        record = {"gpu_uuid": scope, "fencing_token": token}
        if lease in self.leases and self.leases[lease] != record:
            raise ValueError("lease_fence_conflict")
        updated = ScopedFenceState(self.cards, self.leases)
        updated.leases[lease] = record
        card = updated.cards.setdefault(scope, {"minimum": 0, "consumed_tokens": []})
        card["minimum"] = max(card["minimum"], token)
        if seal:
            card["consumed_tokens"] = sorted(set(card["consumed_tokens"]) | {token})
        return updated

    @classmethod
    def load(cls, payload, *, grants, revoked, consumed, minimum, legacy_records=None):
        if any(token > minimum for token in consumed):
            raise ValueError("consumed_above_aggregate_minimum")
        version = payload.get("schema_version", 1)
        if type(version) is not int or version not in (1, 2):
            raise ValueError("unsupported_fence_schema")
        if version == 2:
            raw = payload.get("scope_state")
            if not isinstance(raw, dict) or type(raw.get("version")) is not int or raw["version"] != 1:
                raise ValueError("invalid_scope_state")
            cards, leases = raw.get("cards"), raw.get("leases")
            if not isinstance(cards, dict) or not isinstance(leases, dict):
                raise ValueError("invalid_scope_state")
            for scope, card in cards.items():
                if scope != "*":
                    gpu_scope(scope)
                if not isinstance(card, dict):
                    raise ValueError("invalid_scope_card")
                positive_token(card.get("minimum"))
                tokens = card.get("consumed_tokens")
                if not isinstance(tokens, list):
                    raise ValueError("invalid_scope_consumed")
                for token in tokens:
                    if positive_token(token) > card["minimum"]:
                        raise ValueError("scope_consumed_above_minimum")
            state = cls(cards, leases)
            for lease, rec in leases.items():
                if not isinstance(rec, dict) or not isinstance(lease, str) or not lease:
                    raise ValueError("invalid_scope_lease")
                token, scope = positive_token(rec.get("fencing_token")), rec.get("gpu_uuid")
                if scope not in cards or cards[scope]["minimum"] < token:
                    raise ValueError("scope_lease_above_minimum")
            if max((c["minimum"] for c in cards.values()), default=0) != minimum:
                raise ValueError("scope_aggregate_minimum_mismatch")
            if {t for c in cards.values() for t in c["consumed_tokens"]} != consumed:
                raise ValueError("scope_aggregate_consumed_mismatch")
            if not revoked.issubset(leases):
                raise ValueError("scope_revocation_unresolved")
            for grant in grants:
                rec = leases.get(grant.get("lease_id"))
                if rec != {"gpu_uuid": gpu_scope(grant.get("gpu_uuid")), "fencing_token": positive_token(grant.get("fencing_token"))}:
                    raise ValueError("scope_grant_conflict")
            return state

        if "scope_state" in payload:
            raise ValueError("scope_state_without_schema")
        state = cls()
        for grant in grants:
            state = state.advance(str(grant.get("lease_id") or ""), positive_token(grant.get("fencing_token")), gpu_scope(grant.get("gpu_uuid")), seal=False)
        if any(card["minimum"] > minimum for card in state.cards.values()):
            raise ValueError("legacy_grant_above_minimum")
        records = legacy_records or []
        if not isinstance(records, list):
            raise ValueError("invalid_legacy_lease_records")
        # Supplemental records must be read from the authoritative ledger, not
        # inferred by matching bare token values to some other lease's grant.
        for record in records:
            if not isinstance(record, dict):
                raise ValueError("invalid_legacy_lease_record")
            lease = str(record.get("lease_id") or "")
            if lease not in revoked:
                raise ValueError("unexpected_legacy_lease_record")
            token = positive_token(record.get("fencing_token"))
            if token not in consumed or token > minimum:
                raise ValueError("legacy_lease_token_not_sealed")
            raw_gpu = record.get("gpu_uuid")
            scope = gpu_scope(raw_gpu) if raw_gpu else "*"
            state = state.advance(lease, token, scope, seal=True)
        if not revoked.issubset(state.leases):
            raise ValueError("legacy_fence_scope_unresolved")
        for lease, rec in list(state.leases.items()):
            token = rec["fencing_token"]
            if token in consumed or lease in revoked:
                if token not in consumed:
                    raise ValueError("legacy_revocation_token_missing")
                state = state.advance(lease, token, rec["gpu_uuid"], seal=True)
        accounted = {t for c in state.cards.values() for t in c["consumed_tokens"]}
        # Unattributed integer records retain their full <= barrier everywhere.
        # Never erase a barrier because an old state lacks its origin.
        unknown = consumed - accounted
        known_min = max((c["minimum"] for c in state.cards.values()), default=0)
        floor = max(unknown | ({minimum} if minimum > known_min else {0}))
        if floor:
            card = state.cards.setdefault("*", {"minimum": 0, "consumed_tokens": []})
            card["minimum"] = max(card["minimum"], floor)
            card["consumed_tokens"] = sorted(set(card["consumed_tokens"]) | unknown)
        return state
