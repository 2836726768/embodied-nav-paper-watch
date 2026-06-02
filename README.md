# Embodied Nav Paper Watch

本地轻量版每日论文雷达，面向具身智能导航方向。抓取和初筛只访问公开 arXiv API；中文精读解读默认通过本机 OpenClaw 调用 `openai-codex/gpt-5.5`，并缓存到本地。

## What It Does

- 每日抓取近 `1-3` 天 arXiv 候选论文，默认类别为 `cs.RO`、`cs.CV`、`cs.AI`、`cs.LG`。
- 用关键词命中和 BM25 风格 token 相关度筛选具身智能导航论文。
- 默认调用 OpenClaw 生成中文标题、中文摘要、研究问题、方法要点、实验信息、注意事项和阅读建议。
- 网页参考 `slam3d-paper-watch` 的静态日报形式，重点展示“解决的问题”、“TL;DR（一句话总结）”、“核心贡献”和“局限性”。
- 生成 Markdown 日报到 `out/YYYY-MM-DD.md`，并把同样内容打印到 stdout，方便 OpenClaw cron announce 推送。
- 同步生成静态网页到 `site/index.html`，采用 Daily Paper Reader 风格的左侧精读/速读队列与右侧论文阅读页。
- 保留历史日报：`site/archive/index.html` 列出全部 `out/*.md`，每一天都有固定页 `site/daily/YYYY-MM-DD/index.html`。
- 严格限定具身智能导航方向：论文需要明确包含导航任务、导航策略、VLN/ObjectNav/PointNav、机器人导航规划等核心信号。

## Run Locally

```bash
cd /workspace/embodied-nav-paper-watch
./scripts/daily_push.sh --dry-run
```

常用参数：

```bash
./scripts/daily_push.sh --days 3 --max-results 120 --max-items 10 --min-score 2.0
```

默认会使用 OpenClaw 中文解读。如果只想用本地规则、不调用模型：

```bash
./scripts/daily_push.sh --no-openclaw
```

打开网页：

```bash
cd /workspace/embodied-nav-paper-watch
python3 -m http.server 8080 --directory site
```

然后访问 `http://127.0.0.1:8080/`。历史归档在 `http://127.0.0.1:8080/archive/`。如果只是本机查看，也可以直接打开 `site/index.html`。

## GitHub Pages

当前项目已准备好 `/docs` 发布目录。推送到 GitHub 后，在仓库设置里选择：

```text
Settings -> Pages -> Deploy from a branch
Branch: main
Folder: /docs
```

你的 GitHub Pages 地址将是：

```text
https://2836726768.github.io/embodied-nav-paper-watch/
```

历史归档地址：

```text
https://2836726768.github.io/embodied-nav-paper-watch/archive/
```

每次重新生成网页后，同步发布目录：

```bash
rm -rf docs
cp -a site docs
touch docs/.nojekyll
```

只生成 Markdown、不生成网页：

```bash
./scripts/daily_push.sh --no-site
```

## OpenClaw Cron

当前项目提供安装脚本：

```bash
cd /workspace/embodied-nav-paper-watch
./scripts/install_openclaw_cron.sh --dry-run
./scripts/install_openclaw_cron.sh
```

默认创建的任务等价于：

```bash
openclaw cron create "0 9 * * *" \
  "Run this local command and send the generated Markdown report as the final reply: cd \"/workspace/embodied-nav-paper-watch\" && ./scripts/daily_push.sh" \
  --name "embodied-nav-paper-watch" \
  --session isolated \
  --tz Asia/Shanghai \
  --announce
```

验证任务：

```bash
openclaw cron list
openclaw cron run embodied-nav-paper-watch --wait --wait-timeout 10m
openclaw cron runs --id embodied-nav-paper-watch --limit 10
```

如果你的 OpenClaw CLI 不在 PATH，请先把 `openclaw` 加入 PATH，再运行安装脚本。

## Configure

编辑 `config.yaml` 即可调整方向词、类别、时间窗口、评分阈值、OpenClaw 模型和网页输出目录。默认重点关注：

- embodied navigation / vision-language navigation / VLN
- object-goal / point-goal / robot navigation
- semantic map / active mapping / spatial reasoning
- Habitat / sim-to-real / navigation policy
