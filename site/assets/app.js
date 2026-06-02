(function () {
  const lang = document.querySelector('[data-lang-toggle]');
  if (lang) {
    lang.addEventListener('click', () => {
      document.body.classList.toggle('show-english');
    });
  }
})();
