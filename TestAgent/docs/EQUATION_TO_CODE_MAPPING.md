# Equation to Code Mapping

This document maps mathematical equations from the TestAgentX paper to their implementations in the codebase.

## Overview

The TestAgentX paper presents several key equations that define the system's behavior. This document provides a complete mapping between these equations and their code implementations.

---

## Layer 1: Code Preprocessing

### Equation (1): Code Encoding

**Paper Equation**:
```
s_vec_i = Encoder_Code(s_i)
```

Where:
- `s_i` = Source code method i
- `s_vec_i` = Vector representation of method i
- `Encoder_Code` = Transformer-based encoder (CodeBERT)

**Implementation**: `src/layer1_preprocessing/code_encoder.py`

```python
class CodeEncoder:
    def encode_method(self, method_signature: str, method_body: str) -> CodeEmbedding:
        """
        Implements Equation (1): s_vec_i = Encoder_Code(s_i)
        
        Args:
            method_signature: Method signature (part of s_i)
            method_body: Method body (part of s_i)
            
        Returns:
            CodeEmbedding with s_vec_i (768-dimensional vector)
        """
        # Combine signature and body
        code_text = f"{method_signature}\n{method_body}"  # s_i
        
        # Tokenize
        inputs = self.tokenizer(code_text, ...)
        
        # Encode using CodeBERT
        with torch.no_grad():
            outputs = self.model(**inputs)
            embedding = outputs.last_hidden_state[:, 0, :]  # s_vec_i
        
        return CodeEmbedding(embedding=embedding.cpu().numpy())
```

**Configuration**:
- `config/default_config.yaml`: `code_encoder.embedding_dim = 768`
- Model: `microsoft/codebert-base`

---

## Layer 2: Test Generation

### Equation (2): RL-based Test Prioritization

**Paper Equation**:
```
Q(s, a) ← Q(s, a) + α[r + γ max Q(s', a') - Q(s, a)]
```

Where:
- `Q(s, a)` = Q-value for state s and action a
- `α` = Learning rate
- `r` = Reward
- `γ` = Discount factor
- `s'` = Next state
- `a'` = Next action

**Implementation**: `src/layer2_test_generation/rl_prioritization_agent.py`

```python
class RLPrioritizationAgent:
    def _train_network(self):
        """
        Implements Equation (2): Q-learning update
        
        Q(s, a) ← Q(s, a) + α[r + γ max Q(s', a') - Q(s, a)]
        """
        # Sample batch from replay memory
        states, actions, rewards, next_states, dones = self._sample_batch()
        
        # Current Q-values: Q(s, a)
        current_q = self.q_network(states, actions)
        
        # Next Q-values: max Q(s', a')
        with torch.no_grad():
            next_q = self.target_network(next_states).max(1)[0]
            # Target: r + γ max Q(s', a')
            target_q = rewards + self.gamma * next_q * (1 - dones)
        
        # Loss: [Q(s, a) - target]²
        loss = F.mse_loss(current_q, target_q)
        
        # Update: Q(s, a) ← Q(s, a) + α * gradient
        self.optimizer.zero_grad()
        loss.backward()  # Compute gradients
        self.optimizer.step()  # Apply update with learning rate α
```

**Configuration**:
- `config/default_config.yaml`:
  - `test_generation.rl_prioritization.learning_rate = 0.001` (α)
  - `test_generation.rl_prioritization.gamma = 0.99` (γ)

---

### Equation (3): Test Generation with LLM

**Paper Equation**:
```
T = LLM(s_vec_i, context, prompt)
```

Where:
- `T` = Generated test cases
- `s_vec_i` = Method embedding
- `context` = Semantic context (commit messages, bug reports)
- `prompt` = Few-shot examples

**Implementation**: `src/layer2_test_generation/llm_test_agent.py`

```python
class LLMTestGenerationAgent:
    def generate_tests(self, method_signature: str, method_source: str, 
                      semantic_context: Dict, num_tests: int = 5) -> List[Dict]:
        """
        Implements Equation (3): T = LLM(s_vec_i, context, prompt)
        
        Args:
            method_signature: Part of s_vec_i
            method_source: Part of s_vec_i
            semantic_context: context (commit messages, etc.)
            num_tests: Number of tests to generate
            
        Returns:
            T (list of generated test cases)
        """
        # Build prompt with context and few-shot examples
        prompt = self._build_prompt(
            method_signature,  # s_vec_i
            method_source,     # s_vec_i
            semantic_context,  # context
            self.few_shot_examples  # prompt
        )
        
        # Call LLM: T = LLM(...)
        response = self.llm_client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature
        )
        
        # Parse generated tests: T
        tests = self._parse_tests(response.choices[0].message.content)
        return tests  # T
```

**Configuration**:
- `config/default_config.yaml`:
  - `test_generation.llm.model_name = "gpt-4"`
  - `test_generation.llm.temperature = 0.7`

---

## Layer 3: Fuzzy Validation

### Equation (6): Contextual Relevance Score

**Paper Equation**:
```
CRS(O_b, O_f) = Sim_sem(O_b, O_f) / MaxSim
```

Where:
- `CRS` = Contextual Relevance Score
- `O_b` = Output from buggy version
- `O_f` = Output from fixed version
- `Sim_sem` = Semantic similarity
- `MaxSim` = Maximum possible similarity

**Implementation**: `src/layer3_fuzzy_validation/fuzzy_assertion_agent.py`

```python
class FuzzyAssertionAgent:
    def _calculate_contextual_relevance_score(self, semantic_similarity: float) -> float:
        """
        Implements Equation (6): CRS(O_b, O_f) = Sim_sem(O_b, O_f) / MaxSim
        
        Args:
            semantic_similarity: Sim_sem(O_b, O_f)
            
        Returns:
            CRS (Contextual Relevance Score)
        """
        # CRS(O_b, O_f) = Sim_sem(O_b, O_f) / MaxSim
        return semantic_similarity / self.max_sim
    
    def _calculate_semantic_similarity(self, emb_buggy: torch.Tensor, 
                                      emb_fixed: torch.Tensor) -> float:
        """
        Calculate Sim_sem(O_b, O_f) using cosine similarity
        
        Args:
            emb_buggy: Embedding of O_b
            emb_fixed: Embedding of O_f
            
        Returns:
            Sim_sem(O_b, O_f)
        """
        # Sim_sem = cosine_similarity(emb(O_b), emb(O_f))
        sim = util.pytorch_cos_sim(emb_buggy.unsqueeze(0), 
                                   emb_fixed.unsqueeze(0))
        return sim.item()  # Sim_sem(O_b, O_f)
```

**Configuration**:
- `config/default_config.yaml`:
  - `fuzzy_validation.max_sim = 1.0` (MaxSim)
  - `fuzzy_validation.threshold = 0.7`

---

### Equation (7): Confidence Score

**Paper Equation**:
```
y_hat_t = sigmoid(W^T * o_vec_t + b)
```

Where:
- `y_hat_t` = Confidence score for test t
- `W` = Learned weight matrix
- `o_vec_t` = Output embedding for test t
- `b` = Bias term
- `sigmoid` = Sigmoid activation function

**Implementation**: `src/layer3_fuzzy_validation/fuzzy_assertion_agent.py`

```python
class FuzzyAssertionAgent:
    def __init__(self, ...):
        # Initialize scoring layer: W and b
        self.scoring_layer = nn.Linear(embedding_dim, 1)  # W^T * x + b
    
    def _calculate_confidence_score(self, embedding: torch.Tensor, 
                                   context: Dict = None) -> float:
        """
        Implements Equation (7): y_hat_t = sigmoid(W^T * o_vec_t + b)
        
        Args:
            embedding: o_vec_t (output embedding)
            context: Additional context
            
        Returns:
            y_hat_t (confidence score)
        """
        with torch.no_grad():
            embedding = embedding.unsqueeze(0).to(self.device)  # o_vec_t
            
            # Linear transformation: W^T * o_vec_t + b
            logits = self.scoring_layer(embedding)
            
            # Sigmoid activation: y_hat_t = sigmoid(...)
            confidence = torch.sigmoid(logits)
            
            return confidence.item()  # y_hat_t
```

**Configuration**:
- `config/default_config.yaml`:
  - `fuzzy_validation.confidence_threshold = 0.6`

---

## Layer 4: Patch Verification

### Equation (8): Trace Difference

**Paper Equation**:
```
Δ_trace = Trace(P_f, t_j) - Trace(P_b, t_j)
```

Where:
- `Δ_trace` = Difference in execution traces
- `P_f` = Fixed/patched version
- `P_b` = Buggy version
- `t_j` = Test case j
- `Trace` = Execution trace function

**Implementation**: `src/layer4_patch_regression/patch_verification_agent.py`

```python
class PatchVerificationAgent:
    def _compare_traces(self, buggy_traces: Dict, patched_traces: Dict) -> Dict:
        """
        Implements Equation (8): Δ_trace = Trace(P_f, t_j) - Trace(P_b, t_j)
        
        Args:
            buggy_traces: Trace(P_b, t_j) for all tests
            patched_traces: Trace(P_f, t_j) for all tests
            
        Returns:
            Dictionary with Δ_trace for each test
        """
        differences = {}
        
        for test_id in buggy_traces:
            if test_id not in patched_traces:
                continue
            
            buggy = buggy_traces[test_id]    # Trace(P_b, t_j)
            patched = patched_traces[test_id]  # Trace(P_f, t_j)
            
            # Calculate Δ_trace
            diff = {
                'line_coverage': {
                    'added': list(set(patched.line_coverage) - set(buggy.line_coverage)),
                    'removed': list(set(buggy.line_coverage) - set(patched.line_coverage))
                },
                'method_calls': {
                    'added': list(set(patched.method_calls) - set(buggy.method_calls)),
                    'removed': list(set(buggy.method_calls) - set(patched.method_calls))
                },
                'execution_time_diff': patched.execution_time - buggy.execution_time,
                'memory_usage_diff': patched.memory_usage - buggy.memory_usage
            }
            
            # Calculate effectiveness score based on Δ_trace
            score = self._calculate_effectiveness_score(diff)
            
            differences[test_id] = {
                'differences': diff,  # Δ_trace
                'effectiveness_score': score
            }
        
        return differences
```

**Configuration**:
- `config/default_config.yaml`:
  - `patch_verification.epsilon = 0.1` (threshold for trace differences)
  - `patch_verification.weights.*` (scoring weights)

---

## Layer 5: Knowledge Graph

### DQN for Graph Navigation

**Paper Equation**:
```
Q(s, a) ← Q(s, a) + α[r + γ max Q(s', a') - Q(s, a)]
```

(Same as Equation 2, but applied to graph navigation)

**Implementation**: `src/layer5_knowledge_graph/graph_navigator.py`

```python
class GraphNavigator:
    def _train_network(self):
        """
        Implements DQN for graph navigation
        
        Q(s, a) ← Q(s, a) + α[r + γ max Q(s', a') - Q(s, a)]
        """
        # Sample batch
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Current Q-values: Q(s, a)
        current_q = self.q_network(states).squeeze()
        
        # Next Q-values: max Q(s', a')
        with torch.no_grad():
            next_q = self.target_network(next_states).squeeze()
            # Target: r + γ max Q(s', a')
            target_q = rewards + (1 - dones) * self.gamma * next_q
        
        # Update Q-network
        loss = self.criterion(current_q, target_q.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Soft update target network: θ' ← τθ + (1-τ)θ'
        self._update_target_network(tau=0.001)
```

**Configuration**:
- `config/default_config.yaml`:
  - `knowledge_graph.rl_navigation.learning_rate = 0.001` (α)
  - `knowledge_graph.rl_navigation.gamma = 0.99` (γ)

---

## Mutation Testing

### Mutation Score Calculation

**Formula**:
```
Mutation Score = (Killed Mutants / Total Valid Mutants) × 100%
```

**Implementation**: `src/mutation_testing/mutation_engine.py`

```python
class MutationEngine:
    def run_mutation_testing(self, ...) -> MutationResult:
        """
        Calculate mutation score
        
        Mutation Score = (Killed / (Total - Timeout - Error)) × 100%
        """
        # Count mutants by status
        killed = sum(1 for m in mutants if m.status == "KILLED")
        timeout = sum(1 for m in mutants if m.status == "TIMEOUT")
        errors = sum(1 for m in mutants if m.status == "ERROR")
        total = len(mutants)
        
        # Calculate mutation score
        valid_mutants = total - timeout - errors
        mutation_score = (killed / valid_mutants * 100) if valid_mutants > 0 else 0.0
        
        return MutationResult(
            mutation_score=mutation_score,
            total_mutants=total,
            killed_mutants=killed,
            ...
        )
```

**Configuration**:
- `config/default_config.yaml`:
  - `mutation_testing.timeout_seconds = 60`
  - `mutation_testing.max_mutants_per_file = 100`

---

## Evaluation Metrics

### Coverage Calculation

**Formula**:
```
Line Coverage = (Covered Lines / Total Lines) × 100%
Branch Coverage = (Covered Branches / Total Branches) × 100%
```

**Implementation**: `src/evaluation/metrics_evaluator.py`

```python
class MetricsEvaluator:
    def _parse_coverage_report(self, xml_file: Path) -> CoverageMetrics:
        """
        Calculate coverage metrics from JaCoCo report
        
        Line Coverage = (Covered / Total) × 100%
        """
        # Parse XML
        counters = self._extract_counters(xml_file)
        
        # Calculate line coverage
        line_covered = counters['LINE']['covered']
        line_total = counters['LINE']['total']
        line_coverage = (line_covered / line_total * 100) if line_total > 0 else 0
        
        # Calculate branch coverage
        branch_covered = counters['BRANCH']['covered']
        branch_total = counters['BRANCH']['total']
        branch_coverage = (branch_covered / branch_total * 100) if branch_total > 0 else 0
        
        return CoverageMetrics(
            line_coverage=line_coverage,
            branch_coverage=branch_coverage,
            ...
        )
```

**Configuration**:
- `config/default_config.yaml`:
  - `evaluation.targets.test_coverage = 89.0` (target from paper)

---

## Summary Table

| Equation | Paper Section | Implementation File | Configuration Key |
|----------|---------------|---------------------|-------------------|
| (1) Code Encoding | 3.3.2 | `code_encoder.py` | `code_encoder.*` |
| (2) Q-Learning | 3.4.1 | `rl_prioritization_agent.py` | `test_generation.rl_prioritization.*` |
| (3) LLM Generation | 3.4.2 | `llm_test_agent.py` | `test_generation.llm.*` |
| (6) CRS | 3.5.1 | `fuzzy_assertion_agent.py` | `fuzzy_validation.*` |
| (7) Confidence | 3.5.2 | `fuzzy_assertion_agent.py` | `fuzzy_validation.*` |
| (8) Trace Diff | 3.6 | `patch_verification_agent.py` | `patch_verification.*` |
| DQN Navigation | 3.7 | `graph_navigator.py` | `knowledge_graph.rl_navigation.*` |

---

## Configuration Files

All constants and hyperparameters are now configurable via:

1. **Default Config**: `config/default_config.yaml`
2. **User Config**: Custom YAML file (override defaults)
3. **Environment Variables**: `TESTAGENTX_*` prefix
4. **Code**: `Config` object in `config/config_loader.py`

### Example: Changing Learning Rate

**Option 1: YAML Config**
```yaml
test_generation:
  rl_prioritization:
    learning_rate: 0.01  # Changed from 0.001
```

**Option 2: Environment Variable**
```bash
export TESTAGENTX_TEST_GENERATION__RL_PRIORITIZATION__LEARNING_RATE=0.01
```

**Option 3: Code**
```python
from config.config_loader import get_config

config = get_config()
config.set('test_generation.rl_prioritization.learning_rate', 0.01)
```

---

## References

- TestAgentX Paper: Sections 3.3-3.7
- Configuration Guide: `docs/CONFIGURATION_GUIDE.md`
- API Documentation: `docs/API_REFERENCE.md`
