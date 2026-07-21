Thanks for answering my questions regarding the "Shorting Problem" and the Mean Reversion drift! I coded up your exact suggestions this weekend—decoupling the portfolios and using the BTC/SPY macro gate—and the results were beautiful.

Dropping the BTC/SPY 200MA cross as a hard cash exit completely solved the `long_only` regime filter issue I was struggling with. It smoothed the equity curve out massively: Max DD collapsed from -21% to -15.5% out-of-sample, and it completely dodged the 2022 correlation spike. 

I've spent months over-engineering the math (massive 22k-feature genesis engines, linear multi-factor rotators, etc. all sitting on my GitHub), but the real edge was clearly in the structural architecture like you suggested.

Since I've essentially maxed out the baseline Carver sizing + dual portfolio setup, I'm curious about the next structural leap. In your models, do you find more leverage in stacking highly uncorrelated asset classes in parallel, or in building deeper macro regime filters that dynamically shift capital weightings between the Trend and MR layers?
