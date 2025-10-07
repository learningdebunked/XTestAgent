"""
Implements Equations (4) and (5) from Section 3.4.2:
FDP(t_j) = E[CoverageGain(t_j) + CrashLikelihood(t_j)]
R = α * MutationScore(T) + β * BranchCoverage(T)

RL agent that learns to prioritize tests based on fault detection potential.
"""

import numpy as np
import gym
from gym import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from typing import List, Dict, Tuple, Any, Optional
from dataclasses import dataclass
import torch
import logging
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class TestMetrics:
    """Metrics for a single test"""
    test_id: str
    coverage_gain: float  # Incremental coverage
    crash_likelihood: float  # Estimated bug detection probability
    execution_time: float  # Seconds (normalized 0-1)
    complexity: float  # Cyclomatic complexity of test (normalized 0-1)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'test_id': self.test_id,
            'coverage_gain': self.coverage_gain,
            'crash_likelihood': self.crash_likelihood,
            'execution_time': self.execution_time,
            'complexity': self.complexity
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TestMetrics':
        """Create from dictionary"""
        return cls(
            test_id=data['test_id'],
            coverage_gain=data['coverage_gain'],
            crash_likelihood=data['crash_likelihood'],
            execution_time=data['execution_time'],
            complexity=data['complexity']
        )

class TestPrioritizationEnv(gym.Env):
    """
    Gym environment for learning test prioritization.
    
    State: Features of current test + remaining tests
    Action: Select next test to execute
    Reward: α * MutationScore + β * BranchCoverage (Equation 5)
    """
    
    def __init__(self, tests: List[TestMetrics], alpha: float = 0.6, 
                 beta: float = 0.4):
        super(TestPrioritizationEnv, self).__init__()
        
        self.tests = tests
        self.alpha = alpha  # Weight for mutation score
        self.beta = beta    # Weight for branch coverage
        
        # State: [current_coverage, time_remaining, test_features...]
        # test_features: [coverage_gain, crash_likelihood, execution_time, complexity]
        state_dim = 2 + 4 * len(tests)  # 2 global + 4 per test
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(state_dim,), dtype=np.float32
        )
        
        # Action: Index of next test to execute
        self.action_space = spaces.Discrete(len(tests))
        
        # Internal state
        self.current_coverage = 0.0
        self.time_remaining = 1.0  # Normalized
        self.executed_tests = set()
        self.mutation_score = 0.0
        self.branch_coverage = 0.0
        
        logger.debug(f"Initialized TestPrioritizationEnv with {len(tests)} tests")
    
    def reset(self) -> np.ndarray:
        """Reset environment for new episode"""
        self.current_coverage = 0.0
        self.time_remaining = 1.0
        self.executed_tests = set()
        self.mutation_score = 0.0
        self.branch_coverage = 0.0
        
        logger.debug("Environment reset")
        return self._get_observation()
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Execute action (run a test).
        
        Args:
            action: Index of the test to execute
            
        Returns:
            Tuple of (observation, reward, done, info)
        """
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action {action}. Must be in {self.action_space}")
            
        test = self.tests[action]
        
        # Check if test already executed
        if action in self.executed_tests:
            # Penalty for redundant selection
            reward = -0.1
            done = len(self.executed_tests) == len(self.tests)
            logger.debug(f"Test {action} already executed. Applying penalty.")
            return self._get_observation(), reward, done, {}
        
        # Execute test (simulate)
        self.executed_tests.add(action)
        
        # Update coverage (incremental)
        self.current_coverage = min(1.0, 
            self.current_coverage + test.coverage_gain
        )
        
        # Update branch coverage (simplified)
        self.branch_coverage = self.current_coverage
        
        # Update mutation score (weighted by crash likelihood)
        self.mutation_score = min(1.0,
            self.mutation_score + test.crash_likelihood
        )
        
        # Update time
        self.time_remaining = max(0.0, self.time_remaining - test.execution_time)
        
        # Compute reward (Equation 5)
        reward = (
            self.alpha * self.mutation_score + 
            self.beta * self.branch_coverage
        )
        
        # Episode ends when all tests executed or time runs out
        done = (
            len(self.executed_tests) == len(self.tests) or 
            self.time_remaining <= 0
        )
        
        info = {
            'mutation_score': self.mutation_score,
            'branch_coverage': self.branch_coverage,
            'tests_executed': len(self.executed_tests),
            'time_remaining': self.time_remaining
        }
        
        logger.debug(f"Executed test {action}. Reward: {reward:.3f}, "
                    f"Mutation: {self.mutation_score:.3f}, "
                    f"Coverage: {self.branch_coverage:.3f}")
        
        return self._get_observation(), float(reward), done, info
    
    def _get_observation(self) -> np.ndarray:
        """Get current state observation"""
        # Global state
        state = [self.current_coverage, self.time_remaining]
        
        # Per-test features
        for i, test in enumerate(self.tests):
            if i in self.executed_tests:
                # Test already executed - zero out features
                state.extend([0.0, 0.0, 0.0, 0.0])
            else:
                state.extend([
                    test.coverage_gain,
                    test.crash_likelihood,
                    test.execution_time,
                    test.complexity
                ])
        
        return np.array(state, dtype=np.float32)
    
    def render(self, mode: str = 'human') -> None:
        """Render current state"""
        if mode == 'human':
            print(f"\n=== Test Prioritization Environment ===")
            print(f"Coverage: {self.current_coverage:.2f}")
            print(f"Mutation Score: {self.mutation_score:.2f}")
            print(f"Tests Executed: {len(self.executed_tests)}/{len(self.tests)}")
            print(f"Time Remaining: {self.time_remaining:.2f}")
            
            print("\nTest Execution Order:")
            for i, test_idx in enumerate(self.executed_tests, 1):
                test = self.tests[test_idx]
                fdp = test.coverage_gain + test.crash_likelihood
                print(f"  {i}. {test.test_id} (FDP: {fdp:.2f})")
            
            print("\nPending Tests:")
            pending = [i for i in range(len(self.tests)) if i not in self.executed_tests]
            for test_idx in pending:
                test = self.tests[test_idx]
                fdp = test.coverage_gain + test.crash_likelihood
                print(f"  - {test.test_id} (FDP: {fdp:.2f}, "
                      f"Time: {test.execution_time:.2f})")


class RLPrioritizationAgent:
    """
    Reinforcement Learning agent for test prioritization.
    Implements Section 3.4.2 from paper.
    """
    
    def __init__(self, alpha: float = 0.6, beta: float = 0.4,
                 model_path: Optional[str] = None):
        """
        Initialize RL agent.
        
        Args:
            alpha: Weight for mutation score (Equation 5)
            beta: Weight for branch coverage (Equation 5)
            model_path: Optional path to load a pre-trained model
        """
        self.alpha = alpha
        self.beta = beta
        self.model = None
        self.is_trained = False
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        
        logger.info(f"Initialized RLPrioritizationAgent (alpha={alpha}, beta={beta})")
    
    def train(self, training_episodes: List[List[TestMetrics]], 
             total_timesteps: int = 10000,
             learning_rate: float = 0.0003,
             batch_size: int = 64) -> None:
        """
        Train RL agent on test prioritization task.
        
        Args:
            training_episodes: List of test suites for training
            total_timesteps: Number of training steps
            learning_rate: Learning rate for the optimizer
            batch_size: Batch size for training
        """
        if not training_episodes:
            raise ValueError("No training episodes provided")
        
        logger.info(f"Starting training with {len(training_episodes)} test suites "
                   f"for {total_timesteps} timesteps")
        
        # Create training environment
        def make_env():
            # Sample random test suite from training data
            tests = np.random.choice(training_episodes)
            return TestPrioritizationEnv(tests, self.alpha, self.beta)
        
        env = DummyVecEnv([make_env])
        
        try:
            # Initialize PPO agent
            self.model = PPO(
                policy="MlpPolicy",
                env=env,
                learning_rate=learning_rate,
                n_steps=2048,
                batch_size=batch_size,
                gamma=0.99,
                verbose=1,
                device='auto',  # Use GPU if available
                tensorboard_log="./logs/rl_prioritization/"
            )
            
            # Train
            logger.info(f"Training RL agent for {total_timesteps} timesteps...")
            self.model.learn(total_timesteps=total_timesteps)
            
            self.is_trained = True
            logger.info("Training complete!")
            
        except Exception as e:
            logger.error(f"Error during training: {str(e)}")
            raise
        
        finally:
            # Clean up
            env.close()
    
    def prioritize_tests(self, tests: List[TestMetrics]) -> List[int]:
        """
        Prioritize tests using trained RL agent.
        
        Computes FDP (Fault Detection Potential) for each test.
        Implements Equation (4): FDP(t_j) = E[CoverageGain + CrashLikelihood]
        
        Args:
            tests: List of tests to prioritize
            
        Returns:
            List of test indices in priority order
        """
        if not tests:
            return []
            
        if not self.is_trained:
            logger.warning("Using heuristic prioritization (model not trained)")
            return self._heuristic_prioritization(tests)
        
        try:
            # Use RL model to determine order
            env = TestPrioritizationEnv(tests, self.alpha, self.beta)
            obs = env.reset()
            
            priority_order = []
            done = False
            
            while not done and len(priority_order) < len(tests):
                # Get action from model
                action, _states = self.model.predict(obs, deterministic=True)
                
                # Execute action
                obs, _, done, _ = env.step(action)
                
                if action not in priority_order:
                    priority_order.append(int(action))
            
            # Add any remaining tests that weren't selected
            remaining = [i for i in range(len(tests)) if i not in priority_order]
            priority_order.extend(remaining)
            
            logger.info(f"Prioritized {len(priority_order)} tests using RL agent")
            return priority_order
            
        except Exception as e:
            logger.error(f"Error in RL prioritization: {str(e)}. Falling back to heuristic.")
            return self._heuristic_prioritization(tests)
    
    def _heuristic_prioritization(self, tests: List[TestMetrics]) -> List[int]:
        """
        Heuristic-based prioritization (fallback).
        Uses FDP formula from Equation (4).
        """
        fdp_scores = []
        
        for i, test in enumerate(tests):
            # FDP(t_j) = E[CoverageGain(t_j) + CrashLikelihood(t_j)]
            fdp = test.coverage_gain + test.crash_likelihood
            fdp_scores.append((i, fdp))
        
        # Sort by FDP (descending), then by execution time (ascending)
        priority_order = [
            idx for idx, _ in sorted(
                fdp_scores, 
                key=lambda x: (-x[1], tests[x[0]].execution_time)
            )
        ]
        
        logger.info(f"Prioritized {len(priority_order)} tests using heuristic")
        return priority_order
    
    def save_model(self, path: str) -> None:
        """
        Save trained model to disk.
        
        Args:
            path: Directory path to save the model
        """
        if not self.model:
            logger.warning("No model to save")
            return
            
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            self.model.save(path)
            logger.info(f"Model saved to {path}")
        except Exception as e:
            logger.error(f"Error saving model to {path}: {str(e)}")
    
    def load_model(self, path: str) -> None:
        """
        Load pre-trained model from disk.
        
        Args:
            path: Path to the saved model
        """
        try:
            self.model = PPO.load(path)
            self.is_trained = True
            logger.info(f"Model loaded from {path}")
        except Exception as e:
            logger.error(f"Error loading model from {path}: {str(e)}")
            self.is_trained = False


def example_usage() -> None:
    """Example usage of the RL prioritization agent"""
    import time
    
    print("Starting RL Prioritization Agent example...")
    start_time = time.time()
    
    try:
        # Create sample tests
        tests = [
            TestMetrics(
                test_id=f"test_{i+1}",
                coverage_gain=0.2 + 0.1 * i,  # Varying coverage
                crash_likelihood=0.1 + 0.15 * i,  # Varying crash likelihood
                execution_time=0.1 * (i + 1),  # Increasing execution time
                complexity=0.3 + 0.1 * i  # Varying complexity
            )
            for i in range(5)
        ]
        
        # Initialize agent
        agent = RLPrioritizationAgent(alpha=0.6, beta=0.4)
        
        # Create training data (list of test suites)
        training_episodes = [tests] * 5  # Simple example with same tests
        
        # Train the agent (in a real scenario, use more diverse training data)
        print("Training agent...")
        agent.train(
            training_episodes=training_episodes,
            total_timesteps=5000,  # Short training for example
            learning_rate=0.0003,
            batch_size=32
        )
        
        # Save the trained model
        os.makedirs("models", exist_ok=True)
        model_path = "models/test_prioritization_agent"
        agent.save_model(model_path)
        
        # Prioritize tests
        print("\nPrioritizing tests...")
        priority_order = agent.prioritize_tests(tests)
        
        # Display results
        print("\nTest Prioritization Order:")
        print("-" * 50)
        print(f"{'Rank':<5} {'Test ID':<10} {'Coverage':<10} "
              f"{'Crash Prob':<12} {'Exec Time':<10} FDP")
        print("-" * 50)
        
        for rank, test_idx in enumerate(priority_order, 1):
            test = tests[test_idx]
            fdp = test.coverage_gain + test.crash_likelihood
            print(f"{rank:<5} {test.test_id:<10} {test.coverage_gain:<10.3f} "
                  f"{test.crash_likelihood:<12.3f} {test.execution_time:<10.3f} {fdp:.3f}")
        
        print("\nExplanation:")
        print("- The agent prioritizes tests with higher Fault Detection Potential (FDP)")
        print("- FDP = Coverage Gain + Crash Likelihood")
        print("- The RL model learns to optimize the trade-off between coverage "
              "and fault detection")
    
    except Exception as e:
        print(f"Error in example: {str(e)}")
        import traceback
        traceback.print_exc()
    
    finally:
        print(f"\nExample completed in {time.time() - start_time:.2f} seconds")


if __name__ == "__main__":
    # Run the example
    example_usage()
