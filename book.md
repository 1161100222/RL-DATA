

### 13.1 Policy Approximation and its advatages 策略近似和优势
- 使用softmax函数使得策略归于determinism， (即便是epsilon-gredy）
- 策略参数化最简单的优势是其可能有一个更简单的近似函数（approximate），更快速地学习并产生渐进最优策略（**superior asymptotic**）
- 便于处理随机最优 （**stochastic optimal**）
- 便于注入先验知识
- 更强的收敛保证，使得**连续**的行为函数优于基于action的价值函数\color{#0F0}{text}

> without losing any meaningful generality 不失一般性

$$J(\theta)=v_{\pi_\theta}(s_0)$$

### 13.2 Policy Gradient Theorem 策略梯度理论
$$\nabla J(\theta) = \sum_{s} {\mu_{\pi}(s) \sum_{a} {q_{\pi}(s,a)} \nabla_{\theta} \pi(a|s,\theta)} $$


加入 *baselines*:  $b(s)$ 
$$\nabla J(\theta) = \sum_{s} {\mu_{\pi}(s) \sum_{a} {(q_{\pi}(s,a)-b(s))} \nabla_{\theta} \pi(a|s,\theta)} $$

**a new version of REINFORCE**
更新规则( *update rule*) 的*variance* 变化巨大( *significant effct*) *方差*
$$\theta_{t+1} \overset {.}{=} \theta_t + \alpha \gamma^t (G_t - b(S_t)) \frac{\nabla_\theta\pi(A_t|S_t,\theta)}{\pi(A_t|S_t,\theta)}. $$ 

$b(s)$的状态价值函数的估计：$\hat{v} (S_t,\textbf{w})$  $\textbf{w} \in \Bbb{R^m}$


:-)
输入： 策略参数 $\pi(a|s, \boldsymbol \theta )$    $\to$ $\boldsymbol \theta$
输入： 状态价值函数 $\hat{v}(s,\bold w) $  $\to$ $\bold{w}$
$G_t \leftarrow$ return from step t   
$\delta \leftarrow G_t - \hat{v}(S_t,\bold{w})$
$\bold w \leftarrow \bold w +\beta \delta \nabla_{\bold w} \hat{v} (S_t,\bold w)$
$\boldsymbol \theta \leftarrow \boldsymbol \theta +\alpha \gamma^t\delta \nabla_{\boldsymbol \theta} log\pi(A_t|S_t,\boldsymbol \theta)$

### 13.5 Actor-Citic Methods
#### a

####

使用baseline的强化学习 **(REINFORCE-with-baselines)** 不是*actor-critic* 方法因为状态函数仅仅使用作为baseline，没有作为*critic*
> That is, it is not used for *bootstrapping* (updating a state from theo estimated values of subsequent states),this is a useful *distinction*, for only through bootstrapping do we introduce bias and an **asymptotic dependence** on <u>the quality of the function approximation</u>

*is often on balance beneficial* 总的来说有益：**reduces variance** and **accelerates learning**
the **bias** introduced through bootstrapping and **reliance** on the state representation

类似于蒙特卡洛方法 缺点：
- 学习速度慢（无偏，高方差）
- 在线学习和连续问题不方便

一步的actor-critic用一步的收益取代全部收益（**REINFORCE**中用到），如下:
$$\boldsymbol \theta_{t+1}  \overset.= \boldsymbol \theta_t + \alpha \gamma^t(G_{t:t+1} - \hat v(S_t,\bold w)) \frac {\nabla_{\boldsymbol\theta} \pi(A_t|S_t,\boldsymbol\theta)}{\pi(A_t|S_t,\boldsymbol\theta)} \tag{1}$$
$$= \boldsymbol \theta_t + \alpha \gamma^t(R_{t+1} + \gamma\hat{v}(S_{t+1,\bold w}) - \hat v(S_t,\bold w)) \frac {\nabla_{\boldsymbol\theta} \pi(A_t|S_t,\boldsymbol\theta)}{\pi(A_t|S_t,\boldsymbol\theta)} \tag{2}$$
$$= \boldsymbol \theta_t + \alpha \gamma^t \delta_t \frac {\nabla_{\boldsymbol\theta} \pi(A_t|S_t,\boldsymbol\theta)}{\pi(A_t|S_t,\boldsymbol\theta)} \tag{3}$$


*一步的方法是完全在线和增量式的*
>The main appeal of one-step methods is that they are fully online and incremental

输入： 策略参数 $\pi(a|s, \boldsymbol \theta )$    $\to$ $\boldsymbol \theta$
输入： 状态价值函数 $\hat{v}(s,\bold w) $  $\to$ $\bold{w}$
参数：步长大小 $\alpha>0, \beta>0$
一直重复(对每一个episode):
&emsp;&emsp;初始化S（第一个状态）
&emsp;&emsp;$I\leftarrow 1$
&emsp;&emsp;while S 非终结: 
&emsp;&emsp;&emsp;&emsp;$A\pi(.|S,\boldsymbol\theta)$
&emsp;&emsp;&emsp;&emsp;根据当前状态和奖励执行动作$A$，观察$S'，R$，
&emsp;&emsp;&emsp;&emsp;$\delta \leftarrow R+\gamma\hat{v}(S',\bold{w}) - \hat{v}(S,\bold{w})$
&emsp;&emsp;&emsp;&emsp;$\bold w \leftarrow \bold w +\beta \delta \nabla_{\bold w} \hat{v} (S_t,\bold w)$
&emsp;&emsp;&emsp;&emsp;$\boldsymbol \theta \leftarrow \boldsymbol \theta +\alpha I \delta \nabla_{\boldsymbol \theta} log\pi(A|S,\boldsymbol \theta)$
&emsp;&emsp;&emsp;&emsp;$I \leftarrow \gamma I$
&emsp;&emsp;&emsp;&emsp;$S \leftarrow S'$

### 13.6 Policy Gradient for Continuing Problems

特殊的定义$\mu_\pi(s): lim_{t\to \infin} Pr\{S_t = s|A_{0:t}~\pi\}$

-
-
-
-
-
-
-
-
-
-
-
-
-
-