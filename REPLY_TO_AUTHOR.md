Thanks for answering my questions regarding the "Shorting Problem" and the Mean Reversion drift! I coded up your exact suggestions this weekend—decoupling the portfolios and using the BTC/SPY macro gate—and the results were beautiful.

Dropping the BTC/SPY 200MA cross as a hard cash exit completely solved the `long_only` regime filter issue. It smoothed the equity curve out massively: Max DD collapsed from -21% to -15.5% out-of-sample, and it completely dodged the 2022 correlation spike. 

I just pushed the entire OCURE-5 Dynamic Equity Engine architecture to my GitHub. I've spent months over-engineering the math (massive 22k-feature genesis engines, linear multi-factor rotators, etc.), but the real edge was clearly in the structural architecture like you suggested. 

Since I've essentially maxed out the baseline Carver sizing + dual portfolio setup (hovering around the 1.0 - 1.1 SR mark), I'd love for you to review the codebase if you have a minute. When you review it, what should I be paying the most attention to? General coding mistakes, or flaws in the mathematical reproduction of the Carver sizing?

Also, to help me break past this current low SR fence, I have a follow-up architectural question: Do you temporarily turn off CSM (Cross-Sectional Momentum) when trading highly correlated equities, or is there an institutional way you force dispersion during crashes?

Any direction on the next structural leap would be massively appreciated.
