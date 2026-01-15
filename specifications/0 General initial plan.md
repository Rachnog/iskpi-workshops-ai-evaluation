Okay, let's outline a lecture on the evolution of AI evaluation methods in finance, specifically investment management. I want to cover these dimensions:

* **Process:** How is the investment management process augmented with AI/ML?
* **Data:** What data is needed to do it well?
* **Algorithm:** The AI/ML model itself.
* **Evaluation:** Specific ways to evaluate the AI/ML approach.
* **Challenges:** What are the challenges with this approach?

The specific task will be goal-based investing. It will be about customer requests that sound like, "I want to save for retirement," "I want to buy a house," or "I want to save for my children's education," etc.

I want to start my lecture by introducing the investment field to mathematics students. I need to show what financial markets are, what public data we can get from them, and what the traditional modeling approaches are—for example, Markowitz portfolios or Fama-French models. I also want to give a hint about more advanced models, like hierarchical portfolio clustering.

From an optimization perspective, regarding specific metrics, I want to explain that we can target pure risk optimization, which can be represented differently using volatility, drawdowns, value at risk, etc. We can also target returns, but there are blended metrics representing risk-adjusted returns, like the Sharpe ratio, Sortino ratio, and other metrics.

Those are measured on historical backtests. But, of course, we can have specific metrics for additional algorithms used for returns prediction or risk estimation. For example, even if we use a vanilla covariance matrix, it's still an estimation that requires numerical methods, and we need to measure it. The same goes for hierarchical portfolio clustering, where we also run an algorithm that has to be measured. Running machine learning models to predict something is self-obvious, but this is just the mathematical core of it.

I need to go broader and say that investment management and asset management are parts of a broader wealth management direction. In wealth management, many investors aren't professionals; they don't understand mathematics, metrics, or machine learning. Their goals are sounded in a much simpler way, and a lot of expert work from financial advisors is needed to translate such narratives into mathematical models.

I'll cover three approaches. First, traditional mathematical modeling and machine learning for portfolio optimization, essentially using mean variance, hierarchical portfolio clustering, maybe with some elements of forecasting. Second, using large language models and LLM agents to help turn human narrative and human description into a mathematical modeling task. And last, using a complete AI agent.

For the first mathematical model approach, I want to show that the process requires a lot of human expertise to translate from the narrative of a goal to an optimization problem. You need to select the universe to quantify constraints, time horizon, desired asset classes, etc. Only then can you run some optimization process, and in the end, you need to explain it to the customer.
In the second case, I want to show how large language models can be used to translate a customer problem into a mathematical modeling framework and then to generate a report from the optimization results.
In the last approach, I want to have an agent equipped with tools like web search and code execution that can reiterate and run a tool of portfolio optimization multiple times on demand based on its reasoning.

Regarding data:
*   For the mathematical model, you only need the historical market data.
*   For the LLM-based approach, you'll need expert examples of successful reports and translations from human narrative to mathematical model and setup.
*   For the agentic example, my thesis is that you need to collect the whole decision lineage and decision graph, which is very complex to collect.

 Algorithm already described above.
 
 Regarding evaluation:
*   **Mathematical modeling:** We can run backtests.+  **Machine learning models:** They can be scored separately.
*   **Large language model application:** We can collect a dataset and design an LLM-as-a-judge approach to evaluate outputs using another LLM. *   **AI Agent application:** Alternatively, we can design an agent that judges another agent.

Specific note about LLMs as judges: I want to use the LangFuse framework in this example to demonstrate how to build evaluators.

Specific note about the AI agent evaluation: I'll use the agent-to-agent protocol, specifically the by-agent bits framework, to create two agents. One will do the work, and the other will evaluate it. They'll communicate with each other to run the evaluation of the task.

The challenge with mathematical modeling is that past performance doesn't guarantee future results. With LLMs, you need a really accurate dataset of correct answers in human language, which is hard to automate. And with AI agents, you ideally need a decision log, which almost no one in companies is storing. This is where the biggest opportunities are today in enterprise AI.

I want to finish this lesson with a brief introduction to more complex concepts. In all the examples above, we were dealing with historical datasets. For mathematical modeling, we used historical market data. For LLM evaluation, we use a dataset of questions, and for AI agent evaluation, we use a dataset of scenarios.

I want to emphasize that in finance, especially when dealing with public financial instruments, we're in a highly stochastic environment. All our modeling done on past data doesn't guarantee future performance. That's why, for mathematical modeling, LLMs, and AI agents, we need to think not only in terms of fixed historical datasets but in terms of dynamic environments.

These dynamic environments can be synthetic or real. If we're talking about mathematical modeling, we need to manage a portfolio in real-time, executing trades and making actual investments, acting dynamically based on market behavior. It's not a static policy or recommendation; it's dynamic. It can and should change over time depending on market conditions and the environment we're acting in.

The same goes for AI agents. They should observe the environment, understand the customer, and evaluate performance according to dynamic market realities.

A couple of important notes about the implementation: I want a minimalistic setup with three approaches built into three Jupyter notebooks. If you need additional files for classes, fine, but try to avoid it as much as you can. If you can use available Python libraries, even better. The more minimalistic the examples, the better.
