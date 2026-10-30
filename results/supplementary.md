# Supplementary material

On the following, additional supplementary material is provided.


### Models training
Training information and optimization data.

| Item | Value |
|---|---|
| Optimiser | &bull; AdamW<br>&bull; lr $=0.0001$<br>&bull; weight decay $=0.01$<br>&bull; batch size $=1$ |
| Maximum epochs | $40$ |
| Inner validation | $15\%$ of each round's training patients, stratified by first grade $\times$ E2, seed $2027$ + round |
| Early stopping | patience $8$ epochs on the inner-validation loss |
| Checkpoint selection | lowest inner-validation loss |
| CV | $5$ folds on construction set |
| Seeds | $0, 1, 2$ for every learned model |

### LNN
Liquid Neural Network implementation and details.

| Item | Value |
|---|---|
| Hidden size (units) | $256$ |
| Mode | `default` (gated closed-form solution) |
| Backbone | $1$ layer $\times$ $128$ units, input = input ‖ hidden = $1281$ |
| Backbone activation | LeCun tanh: $1.7159 · \tanh(0.666 · x)$ |
| Backbone dropout | $0.0$ |
| Mixed memory / output projection | no / none (`fc` = identity, output = $256$-d hidden state) |
| Initial state | $h_0=0$ |
| Trajectory projector (head) | Linear($256\rightarrow64$) $\rightarrow$ ReLU $\rightarrow$ Dropout($0.2$) $\rightarrow$ Linear($64\rightarrow1$) |

### RNN, LSTM, Neural Operator
Backbone ablation implementation and details.

| Model | Architecture |
|---|---|
| RNN | &bull; $h_0$ = Linear($1025\rightarrow 512$)(x)<br>&bull; `nn.RNNCell(1, 512)` ($\tanh$)<br>&bull; input at each $0.5$-y step = previous prediction<br>&bull; *head*: Linear($512\rightarrow64$) $\rightarrow$ ReLU $\rightarrow$ Dropout($0.2$) $\rightarrow$ Linear($64\rightarrow1$) |
| LSTM | as RNN with `nn.LSTMCell(1, 512)`, $c_0 = 0$ |
| NO (DeepONet) | &bull; *branch*: Linear($1025\rightarrow 256$) $\rightarrow$ ReLU $\rightarrow$ Linear($256\rightarrow 128$)<br>&bull; *trunk*: Linear($1\rightarrow 64$) $\rightarrow$ ReLU $\rightarrow$ Linear($64\rightarrow 128$)<br>&bull; $\hat{y}(t) = \langle\mathrm{branch}(x), \mathrm{trunk}(t)\rangle$; no dropout |

### Baselines
Baseline models implementation and details.

| Baseline | Inputs | Model and settings | Output |
|---|---|---|---|
| First grade alone (FG) | first-visit grade | &bull; no fitting<br>&bull; flat forecast $\hat{y}(t) =$ grade at the first visit | trajectory |
| Cox | &bull; first grade<br>&bull; $k$ PCA of first-visit MinMax(LE‖RE) (PCA on training patients not late) | &bull; `CoxPHFitter` from `lifelines`<br>&bull; L2 (ridge) penalty $0.1$ (`l1_ratio` $= 0$)<br>&bull; $k \in 0–20$ chosen by inner $5$-fold CV (`StratifiedKFold` on late AMD event, seed $2027$) on Harrell's C| $1 − S(t)$ |
| Logistic regression (LR) | &bull; first grade<br>&bull; $10$ PCA components, `StandardScaler` | &bull; `LogisticRegression` from `scikit-learn`<br>&bull; L2, C = 1, `lbfgs`, `max_iter` $=5000$<br>&bull; one model per late AMD event by horizon ($2, 5$ y) | P(late by h) |
| Linear SVM | MinMax(LE‖RE) | &bull; `SVC(kernel='linear')` defaults ($C = 1$)<br>&bull; samples = every starting visit with grade $< 4$ having a graded visit within $\pm0.5$ y of start + h (label = that grade $≥ 4$)<br>&bull; sigmoid (Platt) calibration, `CalibratedClassifierCV` with `GroupKFold(5)` by patient | calibrated P(late by h) |

### Training
Information on hardware and software training.

| Item | Value |
|---|---|
| GPUs | In parallel:<br>&bull; NVIDIA RTX A4500 (127 runs)<br>&bull; NVIDIA GeForce RTX 5080 (107 runs) |
| Software | &bull; Python *3.11.15*<br>&bull; PyTorch *2.11.0+cu128*<br>&bull; `ncps` *1.0.1*<br>&bull; `lifelines` *0.30.0*<br>&bull; `scikit-learn` *1.8.0* |
| Runtime | $45.3h$ |

### Concept prompts
One prompt per concept, in Chinese (as RET-CLIP text tower is `RoBERTa-wwm-ext-base-chinese`).

| Concept | Prompt |
|---|---|
| normal fundus | 正常眼底 |
| drusen | 玻璃膜疣 |
| large drusen | 大玻璃膜疣 |
| pigmentary abnormality | 色素异常 |
| geographic atrophy | 地图样萎缩 |
| choroidal neovascularization | 脉络膜新生血管 |
| macular degeneration | 黄斑变性 |
