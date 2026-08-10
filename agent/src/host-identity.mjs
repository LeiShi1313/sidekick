const BANK_ID_RE = /^[A-Za-z0-9][A-Za-z0-9:_.%-]{0,255}$/;
const TELEGRAM_MATRIX_BRIDGE_ACTOR_RE =
  /^telegram:matrix-bridge:[1-9][0-9]*%3A-?[0-9]+%3A[a-f0-9]{32}$/;

export function isBankId(value) {
  return typeof value === "string" && BANK_ID_RE.test(value);
}

export function isHostIdentity(value) {
  return (
    isBankId(value) &&
    (/:(?:user|channel):/.test(value) ||
      TELEGRAM_MATRIX_BRIDGE_ACTOR_RE.test(value))
  );
}

export function isRequesterCustomizationPrincipal(value) {
  return (
    isHostIdentity(value) &&
    (value.match(/:user:/g) ?? []).length === 1 &&
    !/:channel:|:matrix-bridge:/.test(value)
  );
}
