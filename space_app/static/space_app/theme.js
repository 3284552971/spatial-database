(() => {
  const STORAGE_KEY = 'ui.theme';

  function getTheme() {
    const saved = (localStorage.getItem(STORAGE_KEY) || '').trim();
    if (saved === 'dark' || saved === 'light') return saved;
    return 'light';
  }

  function setTheme(theme) {
    const t = (theme === 'dark') ? 'dark' : 'light';
    document.documentElement.setAttribute('data-theme', t);
    try { localStorage.setItem(STORAGE_KEY, t); } catch (e) {}
    syncToggleLabels();
  }

  function toggleTheme() {
    const cur = getTheme();
    setTheme(cur === 'dark' ? 'light' : 'dark');
  }

  function labelFor(theme) {
    return theme === 'dark' ? '浅色' : '暗色';
  }

  function syncToggleLabels() {
    const theme = getTheme();
    const btns = document.querySelectorAll('[data-theme-toggle]');
    btns.forEach((btn) => {
      try {
        btn.textContent = labelFor(theme);
        btn.setAttribute('aria-label', `切换主题（当前：${theme === 'dark' ? '暗色' : '浅色'}）`);
      } catch (e) {}
    });
  }

  // init
  setTheme(getTheme());

  document.addEventListener('click', (ev) => {
    const t = ev.target;
    const btn = t && t.closest ? t.closest('[data-theme-toggle]') : null;
    if (!btn) return;
    ev.preventDefault();
    toggleTheme();
  });

  document.addEventListener('DOMContentLoaded', () => {
    syncToggleLabels();
  });
})();

