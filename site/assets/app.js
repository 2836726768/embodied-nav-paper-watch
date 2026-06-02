(function () {
  const input = document.querySelector('[data-static-chat-input]');
  if (input) {
    input.addEventListener('focus', () => {
      input.placeholder = '当前是本地静态阅读页；问答入口可后续接 OpenClaw。';
    });
  }
})();
