import { defineConfig } from 'vitepress'

export default defineConfig({
  title: 'ms-swift 源码解析',
  description: '大模型微调框架的深度技术剖析 — 损失函数、超参数、RLHF 算法全覆盖',
  lang: 'zh-CN',

  head: [
    ['link', { rel: 'icon', href: '/favicon.svg' }],
    ['meta', { name: 'theme-color', content: '#646cff' }],
  ],

  themeConfig: {
    logo: '/logo.svg',
    nav: [
      { text: '开始阅读', link: '/chapters/01-what-is-ms-swift' },
      { text: '目录', link: '/contents' },
      { text: '损失函数', link: '/chapters/12-loss-base' },
      { text: 'RLHF 算法', link: '/chapters/18-rlhf-overview' },
      {
        text: '链接',
        items: [
          { text: 'GitHub 源码', link: 'https://github.com/modelscope/ms-swift' },
          { text: 'ModelScope', link: 'https://modelscope.cn' },
        ]
      }
    ],

    sidebar: [
      {
        text: '第一部分：宏观认知',
        items: [
          { text: '第 1 章 ms-swift 是什么', link: '/chapters/01-what-is-ms-swift' },
          { text: '第 2 章 仓库全景与技术栈', link: '/chapters/02-repo-overview' },
          { text: '第 3 章 快速入门', link: '/chapters/03-quick-start' },
        ]
      },
      {
        text: '第二部分：基础设施层',
        items: [
          { text: '第 4 章 参数系统', link: '/chapters/04-arguments' },
          { text: '第 5 章 模型注册系统', link: '/chapters/05-model-registry' },
          { text: '第 6 章 模板系统', link: '/chapters/06-template-system' },
          { text: '第 7 章 数据集系统', link: '/chapters/07-dataset-system' },
          { text: '第 8 章 Tuner 微调方法', link: '/chapters/08-tuners' },
        ]
      },
      {
        text: '第三部分：训练核心',
        items: [
          { text: '第 9 章 Trainer 架构', link: '/chapters/09-trainer' },
          { text: '第 10 章 训练流水线', link: '/chapters/10-pipelines' },
          { text: '第 11 章 推理引擎', link: '/chapters/11-infer-engine' },
        ]
      },
      {
        text: '第四部分：损失函数体系 ⭐',
        items: [
          { text: '第 12 章 BaseLoss 基础架构', link: '/chapters/12-loss-base' },
          { text: '第 13 章 CrossEntropyLoss 详解', link: '/chapters/13-loss-cross-entropy' },
          { text: '第 14 章 嵌入学习损失函数', link: '/chapters/14-loss-embedding' },
          { text: '第 15 章 重排损失函数', link: '/chapters/15-loss-reranker' },
          { text: '第 16 章 Loss Scale 体系', link: '/chapters/16-loss-scale' },
          { text: '第 17 章 Agent Loss Scale', link: '/chapters/17-loss-scale-agent' },
        ]
      },
      {
        text: '第五部分：RLHF 对齐训练 ⭐',
        items: [
          { text: '第 18 章 RLHF 概论与算法对比', link: '/chapters/18-rlhf-overview' },
          { text: '第 19 章 DPO 及其 14 种变体', link: '/chapters/19-dpo' },
          { text: '第 20 章 CPO / SimPO', link: '/chapters/20-cpo-simpo' },
          { text: '第 21 章 ORPO', link: '/chapters/21-orpo' },
          { text: '第 22 章 KTO', link: '/chapters/22-kto' },
          { text: '第 23 章 PPO', link: '/chapters/23-ppo' },
          { text: '第 24 章 GRPO', link: '/chapters/24-grpo' },
          { text: '第 25 章 GKD 知识蒸馏', link: '/chapters/25-gkd' },
          { text: '第 26 章 Reward Model 训练', link: '/chapters/26-rm' },
          { text: '第 27 章 奖励函数体系', link: '/chapters/27-rewards' },
        ]
      },
      {
        text: '第六部分：超参数详解',
        items: [
          { text: '第 28 章 学习率与优化器', link: '/chapters/28-learning-rate-optimizer' },
          { text: '第 29 章 RLHF 超参数详解', link: '/chapters/29-rlhf-hyperparams' },
          { text: '第 30 章 GRPO / PPO 超参数', link: '/chapters/30-grpo-ppo-hyperparams' },
          { text: '第 31 章 部署与推理超参数', link: '/chapters/31-deploy-hyperparams' },
        ]
      },
      {
        text: '第七部分：高级主题',
        items: [
          { text: '第 32 章 DeepSpeed / FSDP', link: '/chapters/32-deepspeed-fsdp' },
          { text: '第 33 章 Megatron 集成', link: '/chapters/33-megatron' },
          { text: '第 34 章 模型导出与部署', link: '/chapters/34-export-deploy' },
          { text: '第 35 章 评估系统', link: '/chapters/35-evaluation' },
        ]
      },
      {
        text: '附录',
        items: [
          { text: '附录 A 超参数参考手册', link: '/chapters/appendix-a-hyperparams' },
          { text: '附录 B 损失函数速查表', link: '/chapters/appendix-b-loss-cheatsheet' },
          { text: '附录 C 术语表', link: '/chapters/appendix-c-glossary' },
          { text: '附录 D 阅读路线图', link: '/chapters/appendix-d-reading-path' },
        ]
      }
    ],

    socialLinks: [
      { icon: 'github', link: 'https://github.com/modelscope/ms-swift' }
    ],

    footer: {
      message: '基于 ms-swift 源码分析生成 | MIT License',
      copyright: 'Copyright © 2024-ms-swift-book contributors'
    },

    search: {
      provider: 'local',
      options: {
        detailedView: true,
      }
    },

    outline: {
      level: [2, 3],
      label: '本页目录'
    }
  }
})
