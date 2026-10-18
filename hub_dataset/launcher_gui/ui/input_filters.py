"""Filtros de teclado para los campos numéricos del Hub (se cargan una vez con demo.load).

- <input type="number"> editables (episodios, FPS, duración, episodio en Curate): todos son
  enteros >= 0, así que solo se aceptan dígitos (el navegador deja escribir 'e', '.', '+', '-').
- Campos con la clase .ip-input (VR headset IP): solo dígitos y puntos.
Bloquea teclas y también lo pegado/arrastrado que traiga otros caracteres.
"""

INPUT_FILTER_JS = """
() => {
    if (window.__pmmInputFilters) return;
    window.__pmmInputFilters = true;

    // Patrón de caracteres permitidos según el campo; null = sin filtro
    const allowed = el => {
        if (!(el instanceof HTMLInputElement || el instanceof HTMLTextAreaElement)) return null;
        if (el.readOnly || el.disabled) return null;
        if (el.type === 'number') return /^[0-9]*$/;
        if (el.closest('.ip-input')) return /^[0-9.]*$/;
        return null;
    };

    document.addEventListener('keydown', e => {
        const re = allowed(e.target);
        if (!re || e.ctrlKey || e.metaKey || e.altKey || e.key.length !== 1) return;
        if (!re.test(e.key)) e.preventDefault();
    }, true);

    // Pegar / arrastrar / autocompletar
    document.addEventListener('beforeinput', e => {
        const re = allowed(e.target);
        if (re && typeof e.data === 'string' && !re.test(e.data)) e.preventDefault();
    }, true);
    document.addEventListener('paste', e => {
        const re = allowed(e.target);
        const txt = (e.clipboardData || window.clipboardData)?.getData('text') ?? '';
        if (re && !re.test(txt.trim())) e.preventDefault();
    }, true);
}
"""
