import { erf } from "mathjs";

// Mathematical constants
/** 1/π, used in volatility approximations */
const FRAC_1_PI = 1 / Math.PI;

/** 1/√2, used in CDF calculations */
const FRAC_1_SQRT_2 = 1 / Math.sqrt(2);

/** 2/√π, used in volatility approximations */
const FRAC_2_SQRT_PI = 2 / Math.sqrt(Math.PI);

/** √2, used in various calculations */
const SQRT_2 = Math.sqrt(2);

/** 1/√(2π), used in the standard normal PDF */
const FRAC_1_SQRT_2PI = 0.3989422804014326779399460599343818684758586311649346576659258296;

/** √(2π), used in volatility approximations */
const SQRT_TWO_PI = (2 * SQRT_2) / FRAC_2_SQRT_PI;

/**
 * Black-Scholes model implementation for option pricing and Greeks calculations.
 * Provides functions to compute option prices, sensitivities (Greeks), and implied volatilities
 * for European call and put options using the Black-Scholes-Merton framework.
 *
 * @remarks
 * This module depends on the `mathjs` library for the error function (`erf`).
 * Ensure `mathjs` is installed (`npm install mathjs`) before using this module.
 *
 * @example
 * ```ts
 * import { BlackScholes } from './black_scholes';
 *
 * const stock = 100;
 * const strike = 100;
 * const rate = 0.05;
 * const sigma = 0.2;
 * const maturity = 1;
 *
 * const callPrice = BlackScholes.call(stock, strike, rate, sigma, maturity);
 * console.log(`Call Price: ${callPrice}`);
 *
 * const greeks = BlackScholes.computeAll(stock, strike, rate, sigma, maturity);
 * console.log(`Call Delta: ${greeks.callDelta}, Put Delta: ${greeks.putDelta}`);
 * ```
 */
namespace BlackScholes {
  /** Custom error for invalid inputs in Black-Scholes calculations */
  class BlackScholesError extends Error {
    constructor(message: string) {
      super(message);
      this.name = "BlackScholesError";
    }
  }

  /**
   * Validates input parameters for Black-Scholes calculations.
   * @param s - Stock price
   * @param k - Strike price
   * @param rate - Risk-free interest rate
   * @param sigma - Volatility
   * @param maturity - Time to maturity in years
   * @throws {BlackScholesError} If any input is invalid
   */
  function validateInputs(
    s: number,
    k: number,
    rate: number,
    sigma: number,
    maturity: number
  ): void {
    if (s < 0) throw new BlackScholesError("Stock price must be non-negative");
    if (k < 0) throw new BlackScholesError("Strike price must be non-negative");
    if (sigma < 0)
      throw new BlackScholesError("Volatility must be non-negative");
    if (maturity < 0)
      throw new BlackScholesError("Maturity must be non-negative");
    if (
      !Number.isFinite(s) ||
      !Number.isFinite(k) ||
      !Number.isFinite(rate) ||
      !Number.isFinite(sigma) ||
      !Number.isFinite(maturity)
    ) {
      throw new BlackScholesError("All inputs must be finite numbers");
    }
  }

  /**
   * Computes the cumulative distribution function (CDF) of the standard normal distribution.
   * @param x - Input value
   * @returns The probability Φ(x) that a standard normal random variable is less than or equal to x
   * @remarks Uses the error function (erf) from mathjs for numerical stability
   */
  function cumNorm(x: number): number {
    return 0.5 + 0.5 * erf(x * FRAC_1_SQRT_2);
  }

  /**
   * Computes the probability density function (PDF) of the standard normal distribution.
   * @param x - Input value
   * @returns The density φ(x) = (1/√(2π)) * exp(-x²/2)
   */
  function incNorm(x: number): number {
    return Math.exp(-0.5 * x * x) * FRAC_1_SQRT_2PI;
  }

  /**
   * Computes the d₁ parameter in the Black-Scholes formula.
   * @param s - Current stock price
   * @param k - Option strike price
   * @param discount - Discount factor (exp(-r * T))
   * @param sqrtMaturitySigma - Volatility times square root of maturity (σ * √T)
   * @returns d₁ = (ln(s / (k * discount)) + 0.5 * σ² * T) / (σ * √T)
   */
  function d1(
    s: number,
    k: number,
    discount: number,
    sqrtMaturitySigma: number
  ): number {
    return (
      Math.log(s / (k * discount)) / sqrtMaturitySigma + 0.5 * sqrtMaturitySigma
    );
  }

  /**
   * Computes the d₂ parameter in the Black-Scholes formula.
   * @param d1Val - Precomputed d₁ value
   * @param sqrtMaturitySigma - Volatility times square root of maturity (σ * √T)
   * @returns d₂ = d₁ - σ * √T
   */
  function d2(d1Val: number, sqrtMaturitySigma: number): number {
    return d1Val - sqrtMaturitySigma;
  }

  /**
   * Returns the maximum of the input value and zero.
   * @param v - Input value
   * @returns v if v > 0, otherwise 0
   */
  function maxOrZero(v: number): number {
    return Math.max(v, 0);
  }

  /**
   * Computes the Black-Scholes call option price with precomputed discount and volatility.
   * @param s - Current stock price
   * @param k - Option strike price
   * @param discount - Discount factor (exp(-r * T))
   * @param sqrtMaturitySigma - Volatility times square root of maturity (σ * √T)
   * @returns The price of the European call option
   * @throws {BlackScholesError} If inputs are invalid
   */
  export function callDiscount(
    s: number,
    k: number,
    discount: number,
    sqrtMaturitySigma: number
  ): number {
    if (s < 0 || k < 0 || discount < 0 || sqrtMaturitySigma < 0) {
      throw new BlackScholesError("Inputs must be non-negative");
    }
    if (sqrtMaturitySigma > 0) {
      const d1Val = d1(s, k, discount, sqrtMaturitySigma);
      const d2Val = d2(d1Val, sqrtMaturitySigma);
      return s * cumNorm(d1Val) - k * discount * cumNorm(d2Val);
    }
    return maxOrZero(s - k);
  }

  /**
   * Computes the Black-Scholes call option price.
   * @param s - Current stock price
   * @param k - Option strike price
   * @param rate - Risk-free interest rate (continuously compounded)
   * @param sigma - Volatility of the stock's returns
   * @param maturity - Time to maturity in years
   * @returns The price of the European call option
   * @throws {BlackScholesError} If inputs are invalid
   * @example
   * ```ts
   * const price = BlackScholes.call(100, 100, 0.05, 0.2, 1);
   * console.log(price); // Approximately 10.45
   * ```
   */
  export function call(
    s: number,
    k: number,
    rate: number,
    sigma: number,
    maturity: number
  ): number {
    validateInputs(s, k, rate, sigma, maturity);
    return callDiscount(
      s,
      k,
      Math.exp(-rate * maturity),
      Math.sqrt(maturity) * sigma
    );
  }

  /**
   * Computes the delta of a Black-Scholes call option.
   * @param s - Current stock price
   * @param k - Option strike price
   * @param rate - Risk-free interest rate
   * @param sigma - Volatility
   * @param maturity - Time to maturity in years
   * @returns The delta (∂C/∂S), sensitivity of the option price to the stock price
   * @throws {BlackScholesError} If inputs are invalid
   */
  export function callDelta(
    s: number,
    k: number,
    rate: number,
    sigma: number,
    maturity: number
  ): number {
    validateInputs(s, k, rate, sigma, maturity);
    const sqrtMaturitySigma = Math.sqrt(maturity) * sigma;
    if (sqrtMaturitySigma > 0) {
      const discount = Math.exp(-rate * maturity);
      return cumNorm(d1(s, k, discount, sqrtMaturitySigma));
    }
    return s > k ? 1 : 0;
  }

  /**
   * Computes the gamma of a Black-Scholes call option.
   * @param s - Current stock price
   * @param k - Option strike price
   * @param rate - Risk-free interest rate
   * @param sigma - Volatility
   * @param maturity - Time to maturity in years
   * @returns The gamma (∂²C/∂S²), second derivative of the option price with respect to the stock price
   * @throws {BlackScholesError} If inputs are invalid
   */
  export function callGamma(
    s: number,
    k: number,
    rate: number,
    sigma: number,
    maturity: number
  ): number {
    validateInputs(s, k, rate, sigma, maturity);
    const sqrtMaturitySigma = Math.sqrt(maturity) * sigma;
    if (sqrtMaturitySigma > 0) {
      const discount = Math.exp(-rate * maturity);
      const d1Val = d1(s, k, discount, sqrtMaturitySigma);
      return incNorm(d1Val) / (s * sqrtMaturitySigma);
    }
    return 0;
  }

  /**
   * Computes the vega of a Black-Scholes call option.
   * @param s - Current stock price
   * @param k - Option strike price
   * @param rate - Risk-free interest rate
   * @param sigma - Volatility
   * @param maturity - Time to maturity in years
   * @returns The vega (∂C/∂σ), sensitivity of the option price to volatility
   * @throws {BlackScholesError} If inputs are invalid
   */
  export function callVega(
    s: number,
    k: number,
    rate: number,
    sigma: number,
    maturity: number
  ): number {
    validateInputs(s, k, rate, sigma, maturity);
    const sqrtMaturitySigma = Math.sqrt(maturity) * sigma;
    if (sqrtMaturitySigma > 0) {
      const discount = Math.exp(-rate * maturity);
      const d1Val = d1(s, k, discount, sqrtMaturitySigma);
      return (s * incNorm(d1Val) * sqrtMaturitySigma) / sigma;
    }
    return 0;
  }

  /**
   * Computes the theta of a Black-Scholes call option.
   * @param s - Current stock price
   * @param k - Option strike price
   * @param rate - Risk-free interest rate
   * @param sigma - Volatility
   * @param maturity - Time to maturity in years
   * @returns The theta (-∂C/∂T), negative rate of change of the option price with respect to time
   * @throws {BlackScholesError} If inputs are invalid
   */
  export function callTheta(
    s: number,
    k: number,
    rate: number,
    sigma: number,
    maturity: number
  ): number {
    validateInputs(s, k, rate, sigma, maturity);
    const sqrtT = Math.sqrt(maturity);
    const sqrtMaturitySigma = sqrtT * sigma;
    if (sqrtMaturitySigma > 0) {
      const discount = Math.exp(-rate * maturity);
      const d1Val = d1(s, k, discount, sqrtMaturitySigma);
      const d2Val = d2(d1Val, sqrtMaturitySigma);
      return (
        -(s * incNorm(d1Val) * sigma) / (2 * sqrtT) -
        rate * k * discount * cumNorm(d2Val)
      );
    }
    return 0;
  }

  /**
   * Computes the rho of a Black-Scholes call option.
   * @param s - Current stock price
   * @param k - Option strike price
   * @param rate - Risk-free interest rate
   * @param sigma - Volatility
   * @param maturity - Time to maturity in years
   * @returns The rho (∂C/∂r), sensitivity of the option price to the interest rate
   * @throws {BlackScholesError} If inputs are invalid
   */
  export function callRho(
    s: number,
    k: number,
    rate: number,
    sigma: number,
    maturity: number
  ): number {
    validateInputs(s, k, rate, sigma, maturity);
    const sqrtMaturitySigma = Math.sqrt(maturity) * sigma;
    if (sqrtMaturitySigma > 0) {
      const discount = Math.exp(-rate * maturity);
      const d1Val = d1(s, k, discount, sqrtMaturitySigma);
      return k * discount * maturity * cumNorm(d2(d1Val, sqrtMaturitySigma));
    }
    return 0;
  }

  /**
   * Computes the Black-Scholes put option price with precomputed discount and volatility.
   * @param s - Current stock price
   * @param k - Option strike price
   * @param discount - Discount factor (exp(-r * T))
   * @param sqrtMaturitySigma - Volatility times square root of maturity (σ * √T)
   * @returns The price of the European put option
   * @throws {BlackScholesError} If inputs are invalid
   */
  export function putDiscount(
    s: number,
    k: number,
    discount: number,
    sqrtMaturitySigma: number
  ): number {
    if (s < 0 || k < 0 || discount < 0 || sqrtMaturitySigma < 0) {
      throw new BlackScholesError("Inputs must be non-negative");
    }
    if (sqrtMaturitySigma > 0) {
      const d1Val = d1(s, k, discount, sqrtMaturitySigma);
      const d2Val = d2(d1Val, sqrtMaturitySigma);
      return k * discount * cumNorm(-d2Val) - s * cumNorm(-d1Val);
    }
    return maxOrZero(k - s);
  }

  /**
   * Computes the Black-Scholes put option price.
   * @param s - Current stock price
   * @param k - Option strike price
   * @param rate - Risk-free interest rate
   * @param sigma - Volatility
   * @param maturity - Time to maturity in years
   * @returns The price of the European put option
   * @throws {BlackScholesError} If inputs are invalid
   */
  export function put(
    s: number,
    k: number,
    rate: number,
    sigma: number,
    maturity: number
  ): number {
    validateInputs(s, k, rate, sigma, maturity);
    return putDiscount(
      s,
      k,
      Math.exp(-rate * maturity),
      Math.sqrt(maturity) * sigma
    );
  }

  /**
   * Computes the delta of a Black-Scholes put option.
   * @param s - Current stock price
   * @param k - Option strike price
   * @param rate - Risk-free interest rate
   * @param sigma - Volatility
   * @param maturity - Time to maturity in years
   * @returns The delta (∂P/∂S), sensitivity of the option price to the stock price
   * @throws {BlackScholesError} If inputs are invalid
   */
  export function putDelta(
    s: number,
    k: number,
    rate: number,
    sigma: number,
    maturity: number
  ): number {
    validateInputs(s, k, rate, sigma, maturity);
    const sqrtMaturitySigma = Math.sqrt(maturity) * sigma;
    if (sqrtMaturitySigma > 0) {
      const discount = Math.exp(-rate * maturity);
      return cumNorm(d1(s, k, discount, sqrtMaturitySigma)) - 1;
    }
    return k > s ? -1 : 0;
  }

  /**
   * Computes the gamma of a Black-Scholes put option.
   * @param s - Current stock price
   * @param k - Option strike price
   * @param rate - Risk-free interest rate
   * @param sigma - Volatility
   * @param maturity - Time to maturity in years
   * @returns The gamma (∂²P/∂S²), identical to call gamma
   * @throws {BlackScholesError} If inputs are invalid
   */
  export function putGamma(
    s: number,
    k: number,
    rate: number,
    sigma: number,
    maturity: number
  ): number {
    return callGamma(s, k, rate, sigma, maturity);
  }

  /**
   * Computes the vega of a Black-Scholes put option.
   * @param s - Current stock price
   * @param k - Option strike price
   * @param rate - Risk-free interest rate
   * @param sigma - Volatility
   * @param maturity - Time to maturity in years
   * @returns The vega (∂P/∂σ), identical to call vega
   * @throws {BlackScholesError} If inputs are invalid
   */
  export function putVega(
    s: number,
    k: number,
    rate: number,
    sigma: number,
    maturity: number
  ): number {
    return callVega(s, k, rate, sigma, maturity);
  }

  /**
   * Computes the theta of a Black-Scholes put option.
   * @param s - Current stock price
   * @param k - Option strike price
   * @param rate - Risk-free interest rate
   * @param sigma - Volatility
   * @param maturity - Time to maturity in years
   * @returns The theta (-∂P/∂T), negative rate of change of the option price with respect to time
   * @throws {BlackScholesError} If inputs are invalid
   */
  export function putTheta(
    s: number,
    k: number,
    rate: number,
    sigma: number,
    maturity: number
  ): number {
    validateInputs(s, k, rate, sigma, maturity);
    const sqrtT = Math.sqrt(maturity);
    const sqrtMaturitySigma = sqrtT * sigma;
    if (sqrtMaturitySigma > 0) {
      const discount = Math.exp(-rate * maturity);
      const d1Val = d1(s, k, discount, sqrtMaturitySigma);
      const d2Val = d2(d1Val, sqrtMaturitySigma);
      return (
        -(s * incNorm(d1Val) * sigma) / (2 * sqrtT) +
        rate * k * discount * cumNorm(-d2Val)
      );
    }
    return 0;
  }

  /**
   * Computes the rho of a Black-Scholes put option.
   * @param s - Current stock price
   * @param k - Option strike price
   * @param rate - Risk-free interest rate
   * @param sigma - Volatility
   * @param maturity - Time to maturity in years
   * @returns The rho (∂P/∂r), sensitivity of the option price to the interest rate
   * @throws {BlackScholesError} If inputs are invalid
   */
  export function putRho(
    s: number,
    k: number,
    rate: number,
    sigma: number,
    maturity: number
  ): number {
    validateInputs(s, k, rate, sigma, maturity);
    const sqrtMaturitySigma = Math.sqrt(maturity) * sigma;
    if (sqrtMaturitySigma > 0) {
      const discount = Math.exp(-rate * maturity);
      const d1Val = d1(s, k, discount, sqrtMaturitySigma);
      return -k * discount * maturity * cumNorm(-d2(d1Val, sqrtMaturitySigma));
    }
    return 0;
  }

  /**
   * Computes the vanna of a Black-Scholes call option.
   * @param s - Current stock price
   * @param k - Option strike price
   * @param rate - Risk-free interest rate
   * @param sigma - Volatility
   * @param maturity - Time to maturity in years
   * @returns The vanna (∂²C/∂S∂σ), sensitivity of delta to volatility
   * @throws {BlackScholesError} If inputs are invalid
   */
  export function callVanna(
    s: number,
    k: number,
    rate: number,
    sigma: number,
    maturity: number
  ): number {
    validateInputs(s, k, rate, sigma, maturity);
    const sqrtMaturitySigma = Math.sqrt(maturity) * sigma;
    if (sqrtMaturitySigma > 0) {
      const discount = Math.exp(-rate * maturity);
      const d1Val = d1(s, k, discount, sqrtMaturitySigma);
      const d2Val = d2(d1Val, sqrtMaturitySigma);
      return -(incNorm(d1Val) * d2Val) / sigma;
    }
    return 0;
  }

  /**
   * Computes the vanna of a Black-Scholes put option.
   * @param s - Current stock price
   * @param k - Option strike price
   * @param rate - Risk-free interest rate
   * @param sigma - Volatility
   * @param maturity - Time to maturity in years
   * @returns The vanna (∂²P/∂S∂σ), identical to call vanna
   * @throws {BlackScholesError} If inputs are invalid
   */
  export function putVanna(
    s: number,
    k: number,
    rate: number,
    sigma: number,
    maturity: number
  ): number {
    return callVanna(s, k, rate, sigma, maturity);
  }

  /**
   * Computes the vomma of a Black-Scholes call option.
   * @param s - Current stock price
   * @param k - Option strike price
   * @param rate - Risk-free interest rate
   * @param sigma - Volatility
   * @param maturity - Time to maturity in years
   * @returns The vomma (∂²C/∂σ²), sensitivity of vega to volatility
   * @throws {BlackScholesError} If inputs are invalid
   */
  export function callVomma(
    s: number,
    k: number,
    rate: number,
    sigma: number,
    maturity: number
  ): number {
    validateInputs(s, k, rate, sigma, maturity);
    const sqrtT = Math.sqrt(maturity);
    const sqrtMaturitySigma = sqrtT * sigma;
    if (sqrtMaturitySigma > 0) {
      const discount = Math.exp(-rate * maturity);
      const d1Val = d1(s, k, discount, sqrtMaturitySigma);
      const d2Val = d2(d1Val, sqrtMaturitySigma);
      return (
        (s * incNorm(d1Val) * d1Val * d2Val * maturity) / sqrtMaturitySigma
      );
    }
    return 0;
  }

  /**
   * Computes the vomma of a Black-Scholes put option.
   * @param s - Current stock price
   * @param k - Option strike price
   * @param rate - Risk-free interest rate
   * @param sigma - Volatility
   * @param maturity - Time to maturity in years
   * @returns The vomma (∂²P/∂σ²), identical to call vomma
   * @throws {BlackScholesError} If inputs are invalid
   */
  export function putVomma(
    s: number,
    k: number,
    rate: number,
    sigma: number,
    maturity: number
  ): number {
    return callVomma(s, k, rate, sigma, maturity);
  }

  /**
   * Computes the charm of a Black-Scholes call option.
   * @param s - Current stock price
   * @param k - Option strike price
   * @param rate - Risk-free interest rate
   * @param sigma - Volatility
   * @param maturity - Time to maturity in years
   * @returns The charm (∂Δ/∂T or ∂²C/∂S∂T), sensitivity of delta to time
   * @throws {BlackScholesError} If inputs are invalid
   */
  export function callCharm(
    s: number,
    k: number,
    rate: number,
    sigma: number,
    maturity: number
  ): number {
    validateInputs(s, k, rate, sigma, maturity);
    const sqrtT = Math.sqrt(maturity);
    const sqrtMaturitySigma = sqrtT * sigma;
    if (sqrtMaturitySigma > 0) {
      const discount = Math.exp(-rate * maturity);
      const d1Val = d1(s, k, discount, sqrtMaturitySigma);
      const d2Val = d2(d1Val, sqrtMaturitySigma);
      return (
        -(incNorm(d1Val) * (2 * rate * maturity - d2Val * sqrtMaturitySigma)) /
        (2 * maturity * sqrtMaturitySigma)
      );
    }
    return 0;
  }

  /**
   * Computes the charm of a Black-Scholes put option.
   * @param s - Current stock price
   * @param k - Option strike price
   * @param rate - Risk-free interest rate
   * @param sigma - Volatility
   * @param maturity - Time to maturity in years
   * @returns The charm (∂Δ/∂T or ∂²P/∂S∂T), identical to call charm
   * @throws {BlackScholesError} If inputs are invalid
   */
  export function putCharm(
    s: number,
    k: number,
    rate: number,
    sigma: number,
    maturity: number
  ): number {
    return callCharm(s, k, rate, sigma, maturity);
  }

  /**
   * Approximates implied volatility using the Corrado-Miller (1996) formula.
   * @param price - Observed option price
   * @param s - Current stock price
   * @param k - Option strike price
   * @param rate - Risk-free interest rate
   * @param maturity - Time to maturity in years
   * @returns Approximate implied volatility
   * @throws {BlackScholesError} If inputs are invalid
   */
  function approximateVol(
    price: number,
    s: number,
    k: number,
    rate: number,
    maturity: number
  ): number {
    validateInputs(s, k, rate, 0, maturity);
    if (price < 0)
      throw new BlackScholesError("Option price must be non-negative");
    const discount = Math.exp(-rate * maturity);
    const x = k * discount;
    const coef = SQRT_TWO_PI / (s + x);
    const helper1 = s - x;
    const c1 = price - 0.5 * helper1;
    const c2 = c1 * c1;
    const c3 = helper1 * helper1 * FRAC_1_PI;
    const bridge1 = c2 - c3;
    const bridgeM = Math.sqrt(Math.max(bridge1, 0));
    return (coef * (c1 + bridgeM)) / Math.sqrt(maturity);
  }

  /**
   * Finds a root using the Newton-Raphson method.
   * @param objFn - Objective function to find root of
   * @param dfn - Derivative of the objective function
   * @param initialGuess - Initial guess for the root
   * @param precision - Convergence precision
   * @param maxIterations - Maximum number of iterations
   * @returns The root if found, or null if convergence fails
   */
  function findRoot(
    objFn: (x: number) => number,
    dfn: (x: number) => number,
    initialGuess: number,
    precision: number,
    maxIterations: number
  ): number | null {
    let x = initialGuess;
    for (let i = 0; i < maxIterations; i++) {
      const fx = objFn(x);
      const dfx = dfn(x);
      if (Math.abs(fx) < precision) return x;
      if (Math.abs(dfx) < 1e-10 || !Number.isFinite(dfx)) return null;
      x -= fx / dfx;
      if (x < 0 || !Number.isFinite(x)) return null;
    }
    return null;
  }

  /**
   * Computes the implied volatility for a call option using an initial guess.
   * @param price - Observed call option price
   * @param s - Current stock price
   * @param k - Option strike price
   * @param rate - Risk-free interest rate
   * @param maturity - Time to maturity in years
   * @param initialGuess - Initial volatility guess
   * @returns The implied volatility, or null if convergence fails
   * @throws {BlackScholesError} If inputs are invalid
   */
  export function callIvGuess(
    price: number,
    s: number,
    k: number,
    rate: number,
    maturity: number,
    initialGuess: number
  ): number | null {
    validateInputs(s, k, rate, 0, maturity);
    if (price < 0)
      throw new BlackScholesError("Option price must be non-negative");
    if (initialGuess < 0)
      throw new BlackScholesError("Initial guess must be non-negative");
    const objFn = (sigma: number) => call(s, k, rate, sigma, maturity) - price;
    const dfn = (sigma: number) => callVega(s, k, rate, sigma, maturity);
    return findRoot(objFn, dfn, initialGuess, 1e-6, 10000);
  }

  /**
   * Computes the implied volatility for a call option.
   * @param price - Observed call option price
   * @param s - Current stock price
   * @param k - Option strike price
   * @param rate - Risk-free interest rate
   * @param maturity - Time to maturity in years
   * @returns The implied volatility, or null if convergence fails
   * @throws {BlackScholesError} If inputs are invalid
   */
  export function callIv(
    price: number,
    s: number,
    k: number,
    rate: number,
    maturity: number
  ): number | null {
    const initialGuess = approximateVol(price, s, k, rate, maturity);
    return callIvGuess(price, s, k, rate, maturity, initialGuess);
  }

  /**
   * Computes the implied volatility for a put option using an initial guess.
   * @param price - Observed put option price
   * @param s - Current stock price
   * @param k - Option strike price
   * @param rate - Risk-free interest rate
   * @param maturity - Time to maturity in years
   * @param initialGuess - Initial volatility guess
   * @returns The implied volatility, or null if convergence fails
   * @throws {BlackScholesError} If inputs are invalid
   */
  export function putIvGuess(
    price: number,
    s: number,
    k: number,
    rate: number,
    maturity: number,
    initialGuess: number
  ): number | null {
    validateInputs(s, k, rate, 0, maturity);
    if (price < 0)
      throw new BlackScholesError("Option price must be non-negative");
    if (initialGuess < 0)
      throw new BlackScholesError("Initial guess must be non-negative");
    const objFn = (sigma: number) => put(s, k, rate, sigma, maturity) - price;
    const dfn = (sigma: number) => putVega(s, k, rate, sigma, maturity);
    return findRoot(objFn, dfn, initialGuess, 1e-6, 10000);
  }

  /**
   * Computes the implied volatility for a put option.
   * @param price - Observed put option price
   * @param s - Current stock price
   * @param k - Option strike price
   * @param rate - Risk-free interest rate
   * @param maturity - Time to maturity in years
   * @returns The implied volatility, or null if convergence fails
   * @throws {BlackScholesError} If inputs are invalid
   */
  export function putIv(
    price: number,
    s: number,
    k: number,
    rate: number,
    maturity: number
  ): number | null {
    validateInputs(s, k, rate, 0, maturity);
    if (price < 0)
      throw new BlackScholesError("Option price must be non-negative");
    const cPrice = price + s - k * Math.exp(-rate * maturity);
    const initialGuess = approximateVol(cPrice, s, k, rate, maturity);
    return putIvGuess(price, s, k, rate, maturity, initialGuess);
  }

  /**
   * Interface for storing option prices and Greeks for both call and put options.
   */
  export interface PricesAndGreeks {
    /** Price of the call option */
    callPrice: number;
    /** Delta of the call option (∂C/∂S) */
    callDelta: number;
    /** Gamma of the call option (∂²C/∂S²) */
    callGamma: number;
    /** Theta of the call option (-∂C/∂T) */
    callTheta: number;
    /** Vega of the call option (∂C/∂σ) */
    callVega: number;
    /** Rho of the call option (∂C/∂r) */
    callRho: number;
    /** Vanna of the call option (∂²C/∂S∂σ) */
    callVanna: number;
    /** Vomma of the call option (∂²C/∂σ²) */
    callVomma: number;
    /** Charm of the call option (∂²C/∂S∂T) */
    callCharm: number;
    /** Price of the put option */
    putPrice: number;
    /** Delta of the put option (∂P/∂S) */
    putDelta: number;
    /** Gamma of the put option (∂²P/∂S²) */
    putGamma: number;
    /** Theta of the put option (-∂P/∂T) */
    putTheta: number;
    /** Vega of the put option (∂P/∂σ) */
    putVega: number;
    /** Rho of the put option (∂P/∂r) */
    putRho: number;
    /** Vanna of the put option (∂²P/∂S∂σ) */
    putVanna: number;
    /** Vomma of the put option (∂²P/∂σ²) */
    putVomma: number;
    /** Charm of the put option (∂²P/∂S∂T) */
    putCharm: number;
  }

  /**
   * Computes all prices and Greeks for call and put options efficiently.
   * @param stock - Current stock price
   * @param strike - Option strike price
   * @param rate - Risk-free interest rate
   * @param sigma - Volatility
   * @param maturity - Time to maturity in years
   * @returns A PricesAndGreeks object containing all prices and Greeks
   * @throws {BlackScholesError} If inputs are invalid
   * @example
   * ```ts
   * const result = BlackScholes.computeAll(100, 100, 0.05, 0.2, 1);
   * console.log(result.callPrice, result.putPrice);
   * ```
   */
  export function computeAll(
    stock: number,
    strike: number,
    rate: number,
    sigma: number,
    maturity: number
  ): PricesAndGreeks {
    validateInputs(stock, strike, rate, sigma, maturity);
    const discount = Math.exp(-rate * maturity);
    const sqrtMaturity = Math.sqrt(maturity);
    const sqrtMaturitySigma = sqrtMaturity * sigma;
    const kDiscount = strike * discount;

    if (sqrtMaturitySigma > 0) {
      const d1Val = d1(stock, strike, discount, sqrtMaturitySigma);
      const d2Val = d2(d1Val, sqrtMaturitySigma);
      const cdfD1 = cumNorm(d1Val);
      const cdfD2 = cumNorm(d2Val);
      const pdfD1 = incNorm(d1Val);
      const commonVega = (stock * pdfD1 * sqrtMaturitySigma) / sigma;

      return {
        callPrice: stock * cdfD1 - kDiscount * cdfD2,
        callDelta: cdfD1,
        callGamma: pdfD1 / (stock * sqrtMaturitySigma),
        callTheta:
          -(stock * pdfD1 * sigma) / (2 * sqrtMaturity) -
          rate * kDiscount * cdfD2,
        callVega: commonVega,
        callRho: kDiscount * maturity * cdfD2,
        callVanna: (commonVega / stock) * (1 - d1Val / sqrtMaturitySigma),
        callVomma: (commonVega * d1Val * d2Val) / sigma,
        callCharm:
          -(pdfD1 * (2 * rate * maturity - d2Val * sqrtMaturitySigma)) /
          (2 * maturity * sqrtMaturitySigma),
        putPrice: stock * (cdfD1 - 1) + kDiscount * (1 - cdfD2),
        putDelta: cdfD1 - 1,
        putGamma: pdfD1 / (stock * sqrtMaturitySigma),
        putTheta:
          -(stock * pdfD1 * sigma) / (2 * sqrtMaturity) +
          rate * kDiscount * (1 - cdfD2),
        putVega: commonVega,
        putRho: -kDiscount * maturity * (1 - cdfD2),
        putVanna: (commonVega / stock) * (1 - d1Val / sqrtMaturitySigma),
        putVomma: (commonVega * d1Val * d2Val) / sigma,
        putCharm:
          -(pdfD1 * (2 * rate * maturity - d2Val * sqrtMaturitySigma)) /
          (2 * maturity * sqrtMaturitySigma),
      };
    }

    return {
      callPrice: maxOrZero(stock - strike),
      callDelta: stock > strike ? 1 : 0,
      callGamma: 0,
      callTheta: 0,
      callVega: 0,
      callRho: 0,
      callVanna: 0,
      callVomma: 0,
      callCharm: 0,
      putPrice: maxOrZero(strike - stock),
      putDelta: strike > stock ? -1 : 0,
      putGamma: 0,
      putTheta: 0,
      putVega: 0,
      putRho: 0,
      putVanna: 0,
      putVomma: 0,
      putCharm: 0,
    };
  }
}

export { BlackScholes };
