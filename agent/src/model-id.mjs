const MODEL_ID_RE = /^[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}$/;

export function isModelId(value) {
  return typeof value === "string" && MODEL_ID_RE.test(value);
}
