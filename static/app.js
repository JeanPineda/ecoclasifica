const input = document.querySelector("#images");
const button = document.querySelector("#classify");
const status = document.querySelector("#status");
const preview = document.querySelector("#preview");
const results = document.querySelector("#results");
let selected = [];
let urls = [];
let lastResults = [];
function element(tag, text, className) {
  const node = document.createElement(tag);
  if (text !== undefined) node.textContent = text;
  if (className) node.className = className;
  return node;
}
input.addEventListener("change", () => {
  urls.forEach(url => URL.revokeObjectURL(url));
  urls = [];
  preview.replaceChildren();
  selected = Array.from(input.files);
  document.querySelector("#results-section").hidden = true;
  const invalid = selected.length > 10 || selected.some(file => file.size > 5 * 1024 * 1024);
  button.disabled = selected.length === 0 || invalid;
  status.textContent = invalid ? "Selecciona hasta 10 fotos de máximo 5 MB cada una." :
    selected.length ? selected.length + " imagen(es) lista(s)." : "Selecciona imágenes para comenzar.";
  selected.forEach(file => {
    const url = URL.createObjectURL(file);
    urls.push(url);
    const img = element("img");
    img.src = url;
    img.alt = file.name;
    img.onerror = () => img.remove();
    preview.append(img);
  });
});
button.addEventListener("click", async () => {
  button.disabled = true;
  input.disabled = true;
  results.replaceChildren();
  lastResults = [];
  document.querySelector("#results-section").hidden = false;
  for (let index = 0; index < selected.length; index++) {
    status.textContent = "Procesando " + (index + 1) + " de " + selected.length + "…";
    const data = new FormData();
    data.append("images", selected[index]);
    let row;
    try {
      const response = await fetch("/predict", { method: "POST", body: data, signal: AbortSignal.timeout(120000) });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.error || "El servicio no pudo procesar la imagen.");
      row = payload.results[0];
    } catch (error) {
      row = { file: selected[index].name, status: "error", error: "No se pudo consultar el servicio. Comprueba la conexión y vuelve a clasificar." };
    }
    lastResults.push(row);
    const card = element("article", undefined, "card");
    const img = element("img");
    img.src = urls[index];
    img.alt = row.file;
    img.onerror = () => img.remove();
    card.append(img, element("p", row.file, "filename"));
    if (row.status === "ok") {
      card.append(element("p", row.label, "label"));
      card.append(element("p", "Puntuación " + (row.score * 100).toFixed(1) + " %"));
      const scores = element("div", undefined, "scores");
      Object.entries(row.scores).forEach(([name, value]) => scores.append(element("div", name + " · " + (value * 100).toFixed(2) + " %")));
      card.append(scores);
    } else card.append(element("p", row.error, "error"));
    results.append(card);
  }
  const ok = lastResults.filter(row => row.status === "ok").length;
  status.textContent = "Finalizado. " + ok + " de " + selected.length + " imagen(es) procesada(s).";
  input.disabled = false;
  button.disabled = false;
});
document.querySelector("#download").addEventListener("click", () => {
  function csvCell(value) {
    let text = String(value ?? "");
    if (/^[=+\-@]/.test(text)) text = "'" + text;
    return '"' + text.replaceAll('"', '""') + '"';
  }
  const rows = [["Archivo", "Estado", "Clase", "Puntuación", "Vidrio", "Metal", "Plástico", "Versión", "Error"]];
  lastResults.forEach(row => rows.push([row.file, row.status, row.label, row.score,
    row.scores?.Vidrio, row.scores?.Metal, row.scores?.["Plástico"], row.model_version, row.error]));
  const blob = new Blob(["\uFEFF" + rows.map(row => row.map(csvCell).join(";")).join("\r\n")], {type:"text/csv;charset=utf-8"});
  const url = URL.createObjectURL(blob);
  const link = element("a"); link.href = url; link.download = "EcoClasifica_resultados.csv"; link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
});
