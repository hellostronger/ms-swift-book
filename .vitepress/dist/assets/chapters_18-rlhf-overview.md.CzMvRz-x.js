import{_ as i,o as p,c as l,a2 as a,j as n,a as t}from"./chunks/framework.DnrXFDHb.js";const C=JSON.parse('{"title":"第 18 章 RLHF 概论与算法对比","description":"","frontmatter":{},"headers":[],"relativePath":"chapters/18-rlhf-overview.md","filePath":"chapters/18-rlhf-overview.md"}'),e={name:"chapters/18-rlhf-overview.md"};function d(h,s,r,k,c,o){return p(),l("div",null,[...s[0]||(s[0]=[a(`<h1 id="第-18-章-rlhf-概论与算法对比" tabindex="-1">第 18 章 RLHF 概论与算法对比 <a class="header-anchor" href="#第-18-章-rlhf-概论与算法对比" aria-label="Permalink to &quot;第 18 章 RLHF 概论与算法对比&quot;">​</a></h1><h2 id="_18-1-对齐技术演进" tabindex="-1">18.1 对齐技术演进 <a class="header-anchor" href="#_18-1-对齐技术演进" aria-label="Permalink to &quot;18.1 对齐技术演进&quot;">​</a></h2><p>大语言模型的对齐技术经历了三个阶段：</p><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>阶段一：SFT（监督微调）</span></span>
<span class="line"><span>   └─ 用人类标注的对话数据训练模型</span></span>
<span class="line"><span>   └─ 问题：可能学到人类的偏见</span></span>
<span class="line"><span></span></span>
<span class="line"><span>阶段二：RLHF（人类反馈强化学习）</span></span>
<span class="line"><span>   └─ 训练奖励模型 + PPO 强化学习</span></span>
<span class="line"><span>   └─ 问题：训练不稳定，超参敏感</span></span>
<span class="line"><span></span></span>
<span class="line"><span>阶段三：DPO 及其变体</span></span>
<span class="line"><span>   └─ 直接在偏好数据上优化</span></span>
<span class="line"><span>   └─ 优势：训练稳定，实现简单</span></span></code></pre></div><h2 id="_18-2-ms-swift-支持的-rlhf-算法" tabindex="-1">18.2 ms-swift 支持的 RLHF 算法 <a class="header-anchor" href="#_18-2-ms-swift-支持的-rlhf-算法" aria-label="Permalink to &quot;18.2 ms-swift 支持的 RLHF 算法&quot;">​</a></h2><table tabindex="0"><thead><tr><th>算法</th><th>类型</th><th>是否需要参考模型</th><th>是否需要奖励模型</th><th>适用场景</th></tr></thead><tbody><tr><td>DPO</td><td>离线偏好优化</td><td>是</td><td>否</td><td>通用对齐</td></tr><tr><td>CPO</td><td>离线偏好优化</td><td>否</td><td>否</td><td>无参考模型</td></tr><tr><td>SimPO</td><td>离线偏好优化</td><td>否</td><td>否</td><td>简化版 CPO</td></tr><tr><td>ORPO</td><td>离线偏好优化</td><td>否</td><td>否</td><td>无需参考模型</td></tr><tr><td>KTO</td><td>离线偏好优化</td><td>是</td><td>否</td><td>二元反馈</td></tr><tr><td>PPO</td><td>在线强化学习</td><td>是</td><td>是</td><td>经典 RLHF</td></tr><tr><td>GRPO</td><td>在线强化学习</td><td>是</td><td>否</td><td>分组相对优化</td></tr><tr><td>GKD</td><td>知识蒸馏</td><td>否</td><td>否</td><td>模型蒸馏</td></tr><tr><td>RM</td><td>奖励模型训练</td><td>否</td><td>-</td><td>训练奖励模型</td></tr></tbody></table><h2 id="_18-3-算法选择指南" tabindex="-1">18.3 算法选择指南 <a class="header-anchor" href="#_18-3-算法选择指南" aria-label="Permalink to &quot;18.3 算法选择指南&quot;">​</a></h2><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>你的场景是什么？</span></span>
<span class="line"><span>│</span></span>
<span class="line"><span>├─ 有偏好数据（chosen/rejected）</span></span>
<span class="line"><span>│  │</span></span>
<span class="line"><span>│  ├─ 想要简单稳定</span></span>
<span class="line"><span>│  │  └─ 选择 DPO（默认推荐）</span></span>
<span class="line"><span>│  │</span></span>
<span class="line"><span>│  ├─ 不想加载参考模型</span></span>
<span class="line"><span>│  │  ├─ 选择 CPO</span></span>
<span class="line"><span>│  │  └─ 选择 SimPO（更简单）</span></span>
<span class="line"><span>│  │</span></span>
<span class="line"><span>│  └─ 数据是二元的（好/坏）</span></span>
<span class="line"><span>│     └─ 选择 KTO</span></span>
<span class="line"><span>│</span></span>
<span class="line"><span>├─ 只有评分数据（无对比）</span></span>
<span class="line"><span>│  └─ 选择 KTO</span></span>
<span class="line"><span>│</span></span>
<span class="line"><span>├─ 想要在线学习</span></span>
<span class="line"><span>│  ├─ 有奖励模型</span></span>
<span class="line"><span>│  │  └─ 选择 PPO</span></span>
<span class="line"><span>│  └─ 无奖励模型</span></span>
<span class="line"><span>│     └─ 选择 GRPO</span></span>
<span class="line"><span>│</span></span>
<span class="line"><span>└─ 想要知识蒸馏</span></span>
<span class="line"><span>   └─ 选择 GKD</span></span></code></pre></div><h2 id="_18-4-核心概念" tabindex="-1">18.4 核心概念 <a class="header-anchor" href="#_18-4-核心概念" aria-label="Permalink to &quot;18.4 核心概念&quot;">​</a></h2><h3 id="参考模型-reference-model" tabindex="-1">参考模型（Reference Model） <a class="header-anchor" href="#参考模型-reference-model" aria-label="Permalink to &quot;参考模型（Reference Model）&quot;">​</a></h3><p>参考模型是 RLHF 中的关键组件，用于约束策略更新幅度：</p>`,11),n("p",null,[t("$$\\mathcal{L}"),n("em",{KL:""},"{KL} = \\beta \\cdot D"),t("(\\pi_\\theta | \\pi_{ref})$$")],-1),a(`<ul><li>DPO/KTO/PPO/GRPO 需要参考模型</li><li>CPO/SimPO/ORPO 不需要参考模型</li></ul><h3 id="奖励模型-reward-model" tabindex="-1">奖励模型（Reward Model） <a class="header-anchor" href="#奖励模型-reward-model" aria-label="Permalink to &quot;奖励模型（Reward Model）&quot;">​</a></h3><p>奖励模型将人类偏好建模为标量分数：</p><p>$$r(x, y) = \\text{RewardModel}(x, y)$$</p><ul><li>PPO 需要奖励模型</li><li>DPO 系列算法不需要奖励模型</li></ul><h3 id="温度参数-temperature" tabindex="-1">温度参数（Temperature） <a class="header-anchor" href="#温度参数-temperature" aria-label="Permalink to &quot;温度参数（Temperature）&quot;">​</a></h3><p>控制策略更新的激进程度：</p><ul><li>温度高：更新更平滑，探索更多</li><li>温度低：更新更保守，利用更多</li></ul><p>默认值：<code>temperature = 0.9</code>（GRPO/PPO/GKD）</p><h2 id="_18-5-训练流程对比" tabindex="-1">18.5 训练流程对比 <a class="header-anchor" href="#_18-5-训练流程对比" aria-label="Permalink to &quot;18.5 训练流程对比&quot;">​</a></h2><h3 id="dpo-训练流程" tabindex="-1">DPO 训练流程 <a class="header-anchor" href="#dpo-训练流程" aria-label="Permalink to &quot;DPO 训练流程&quot;">​</a></h3><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>1. 加载偏好数据 (chosen, rejected)</span></span>
<span class="line"><span>2. 加载策略模型和参考模型</span></span>
<span class="line"><span>3. 对于每个 batch:</span></span>
<span class="line"><span>   - 计算策略模型的 log probabilities</span></span>
<span class="line"><span>   - 计算参考模型的 log probabilities</span></span>
<span class="line"><span>   - 计算 DPO loss</span></span>
<span class="line"><span>   - 反向传播，更新策略模型</span></span></code></pre></div><h3 id="ppo-训练流程" tabindex="-1">PPO 训练流程 <a class="header-anchor" href="#ppo-训练流程" aria-label="Permalink to &quot;PPO 训练流程&quot;">​</a></h3><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>1. 加载 prompt 数据</span></span>
<span class="line"><span>2. 加载策略模型、参考模型、奖励模型、价值模型</span></span>
<span class="line"><span>3. 对于每个 batch:</span></span>
<span class="line"><span>   - 策略模型生成 responses</span></span>
<span class="line"><span>   - 奖励模型计算 rewards</span></span>
<span class="line"><span>   - 价值模型计算 advantages</span></span>
<span class="line"><span>   - 使用 PPO 算法更新策略模型</span></span></code></pre></div><h3 id="grpo-训练流程" tabindex="-1">GRPO 训练流程 <a class="header-anchor" href="#grpo-训练流程" aria-label="Permalink to &quot;GRPO 训练流程&quot;">​</a></h3><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>1. 加载 prompt 数据</span></span>
<span class="line"><span>2. 加载策略模型和参考模型</span></span>
<span class="line"><span>3. 对于每个 batch:</span></span>
<span class="line"><span>   - 对每个 prompt 生成 G 个 responses</span></span>
<span class="line"><span>   - 计算每个 response 的 rewards</span></span>
<span class="line"><span>   - 在组内计算相对优势</span></span>
<span class="line"><span>   - 使用 GRPO 算法更新策略模型</span></span></code></pre></div><h2 id="_18-6-性能对比" tabindex="-1">18.6 性能对比 <a class="header-anchor" href="#_18-6-性能对比" aria-label="Permalink to &quot;18.6 性能对比&quot;">​</a></h2><table tabindex="0"><thead><tr><th>算法</th><th>训练速度</th><th>显存占用</th><th>稳定性</th><th>效果</th></tr></thead><tbody><tr><td>DPO</td><td>快</td><td>中（2倍模型）</td><td>高</td><td>好</td></tr><tr><td>CPO</td><td>快</td><td>低（1倍模型）</td><td>高</td><td>好</td></tr><tr><td>SimPO</td><td>快</td><td>低（1倍模型）</td><td>高</td><td>好</td></tr><tr><td>ORPO</td><td>快</td><td>低（1倍模型）</td><td>高</td><td>好</td></tr><tr><td>KTO</td><td>中</td><td>中（2倍模型）</td><td>中</td><td>中</td></tr><tr><td>PPO</td><td>慢</td><td>高（4倍模型）</td><td>低</td><td>最好</td></tr><tr><td>GRPO</td><td>中</td><td>中（2倍模型）</td><td>中</td><td>好</td></tr><tr><td>GKD</td><td>中</td><td>中（2倍模型）</td><td>高</td><td>好</td></tr></tbody></table><h2 id="_18-7-实际使用示例" tabindex="-1">18.7 实际使用示例 <a class="header-anchor" href="#_18-7-实际使用示例" aria-label="Permalink to &quot;18.7 实际使用示例&quot;">​</a></h2><div class="language-bash vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">bash</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;"># DPO 训练</span></span>
<span class="line"><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">swift</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> rlhf</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> \\</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">  --rlhf_type</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> dpo</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> \\</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">  --model</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> Qwen/Qwen2.5-7B-Instruct</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> \\</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">  --dataset</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> Anthropic/hh-rlhf</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> \\</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">  --beta</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 0.1</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> \\</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">  --output_dir</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> output/dpo</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;"># GRPO 训练</span></span>
<span class="line"><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">swift</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> rlhf</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> \\</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">  --rlhf_type</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> grpo</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> \\</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">  --model</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> Qwen/Qwen2.5-7B-Instruct</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> \\</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">  --dataset</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> openai/gsm8k</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> \\</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">  --reward_funcs</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> accuracy</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> format</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> \\</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">  --num_generations</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 8</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> \\</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">  --beta</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 0.04</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> \\</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">  --output_dir</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> output/grpo</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;"># CPO 训练（无需参考模型）</span></span>
<span class="line"><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">swift</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> rlhf</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> \\</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">  --rlhf_type</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> cpo</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> \\</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">  --model</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> Qwen/Qwen2.5-7B-Instruct</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> \\</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">  --dataset</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> Anthropic/hh-rlhf</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> \\</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">  --cpo_alpha</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 1.0</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> \\</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">  --output_dir</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> output/cpo</span></span></code></pre></div><h2 id="小结" tabindex="-1">小结 <a class="header-anchor" href="#小结" aria-label="Permalink to &quot;小结&quot;">​</a></h2><ul><li><strong>要点 1</strong>：ms-swift 支持 9 种 RLHF 算法，覆盖离线/在线/蒸馏全部场景</li><li><strong>要点 2</strong>：DPO 是最简单稳定的选择，适合大多数场景</li><li><strong>要点 3</strong>：GRPO/PPO 适合在线学习，GRPO 不需要奖励模型</li><li><strong>要点 4</strong>：选择算法时考虑数据格式、计算资源、训练稳定性</li></ul>`,22)])])}const g=i(e,[["render",d]]);export{C as __pageData,g as default};
