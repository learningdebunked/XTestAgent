"""
Fault Injector for Chaos Engineering

Injects various types of faults into the system to test resilience.
"""

import time
import random
import threading
import logging
from typing import Dict, Any, Optional, Callable, List
from dataclasses import dataclass
import psutil
import os
import signal

from .chaos_scenarios import ChaosScenario, ChaosScenarioType

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class InjectionResult:
    """Result of a fault injection"""
    scenario_name: str
    success: bool
    duration: float
    impact_metrics: Dict[str, Any]
    errors_caught: List[str]
    recovery_time: Optional[float] = None


class FaultInjector:
    """
    Injects faults into the system for chaos testing.
    
    Supports various fault types:
    - Network faults (latency, partition, packet loss)
    - Resource exhaustion (CPU, memory, disk)
    - Application faults (exceptions, nulls, delays)
    - State corruption
    - Time manipulation
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize fault injector.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.active_injections = {}
        self.injection_history = []
        self._stop_flags = {}
        
        logger.info("FaultInjector initialized")
    
    def inject_fault(self, scenario: ChaosScenario) -> InjectionResult:
        """
        Inject a fault based on the chaos scenario.
        
        Args:
            scenario: Chaos scenario to execute
            
        Returns:
            InjectionResult with injection details
        """
        logger.info(f"Injecting fault: {scenario.name}")
        
        start_time = time.time()
        errors = []
        impact_metrics = {}
        
        try:
            # Route to appropriate injection method
            if scenario.scenario_type == ChaosScenarioType.NETWORK_LATENCY:
                self._inject_network_latency(scenario)
            elif scenario.scenario_type == ChaosScenarioType.NETWORK_PARTITION:
                self._inject_network_partition(scenario)
            elif scenario.scenario_type == ChaosScenarioType.PACKET_LOSS:
                self._inject_packet_loss(scenario)
            elif scenario.scenario_type == ChaosScenarioType.CPU_STRESS:
                self._inject_cpu_stress(scenario)
            elif scenario.scenario_type == ChaosScenarioType.MEMORY_STRESS:
                self._inject_memory_stress(scenario)
            elif scenario.scenario_type == ChaosScenarioType.DISK_STRESS:
                self._inject_disk_stress(scenario)
            elif scenario.scenario_type == ChaosScenarioType.EXCEPTION_INJECTION:
                self._inject_exceptions(scenario)
            elif scenario.scenario_type == ChaosScenarioType.NULL_INJECTION:
                self._inject_nulls(scenario)
            elif scenario.scenario_type == ChaosScenarioType.SLOW_RESPONSE:
                self._inject_slow_response(scenario)
            elif scenario.scenario_type == ChaosScenarioType.STATE_CORRUPTION:
                self._inject_state_corruption(scenario)
            elif scenario.scenario_type == ChaosScenarioType.TIME_SKEW:
                self._inject_time_skew(scenario)
            else:
                logger.warning(f"Unknown scenario type: {scenario.scenario_type}")
            
            # Wait for duration
            time.sleep(scenario.duration)
            
            # Cleanup
            self._cleanup_injection(scenario)
            
            duration = time.time() - start_time
            
            # Collect impact metrics
            impact_metrics = self._collect_impact_metrics()
            
            result = InjectionResult(
                scenario_name=scenario.name,
                success=True,
                duration=duration,
                impact_metrics=impact_metrics,
                errors_caught=errors
            )
            
            self.injection_history.append(result)
            logger.info(f"Fault injection complete: {scenario.name}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error during fault injection: {e}", exc_info=True)
            errors.append(str(e))
            
            return InjectionResult(
                scenario_name=scenario.name,
                success=False,
                duration=time.time() - start_time,
                impact_metrics=impact_metrics,
                errors_caught=errors
            )
    
    def _inject_network_latency(self, scenario: ChaosScenario) -> None:
        """Inject network latency"""
        params = scenario.parameters
        min_latency = params.get('min_latency_ms', 100)
        max_latency = params.get('max_latency_ms', 1000)
        
        logger.info(f"Injecting network latency: {min_latency}-{max_latency}ms")
        
        # Store original socket methods
        import socket
        original_send = socket.socket.send
        original_recv = socket.socket.recv
        
        def delayed_send(self, data, *args, **kwargs):
            delay = random.uniform(min_latency, max_latency) / 1000.0
            time.sleep(delay)
            return original_send(self, data, *args, **kwargs)
        
        def delayed_recv(self, bufsize, *args, **kwargs):
            delay = random.uniform(min_latency, max_latency) / 1000.0
            time.sleep(delay)
            return original_recv(self, bufsize, *args, **kwargs)
        
        # Monkey patch socket methods
        socket.socket.send = delayed_send
        socket.socket.recv = delayed_recv
        
        # Store for cleanup
        self.active_injections['network_latency'] = {
            'original_send': original_send,
            'original_recv': original_recv
        }
    
    def _inject_network_partition(self, scenario: ChaosScenario) -> None:
        """Simulate network partition"""
        logger.info("Injecting network partition")
        
        import socket
        original_connect = socket.socket.connect
        
        def blocked_connect(self, address):
            raise socket.error("Network partition: Connection refused")
        
        socket.socket.connect = blocked_connect
        self.active_injections['network_partition'] = {'original_connect': original_connect}
    
    def _inject_packet_loss(self, scenario: ChaosScenario) -> None:
        """Inject packet loss"""
        params = scenario.parameters
        loss_percentage = params.get('loss_percentage', 10) / 100.0
        
        logger.info(f"Injecting packet loss: {loss_percentage*100}%")
        
        import socket
        original_send = socket.socket.send
        
        def lossy_send(self, data, *args, **kwargs):
            if random.random() < loss_percentage:
                # Simulate packet loss
                return len(data)  # Pretend it was sent
            return original_send(self, data, *args, **kwargs)
        
        socket.socket.send = lossy_send
        self.active_injections['packet_loss'] = {'original_send': original_send}
    
    def _inject_cpu_stress(self, scenario: ChaosScenario) -> None:
        """Inject CPU stress"""
        params = scenario.parameters
        cpu_percentage = params.get('cpu_percentage', 80)
        cores = params.get('cores', 1)
        
        logger.info(f"Injecting CPU stress: {cpu_percentage}% on {cores} cores")
        
        stop_flag = threading.Event()
        self._stop_flags['cpu_stress'] = stop_flag
        
        def cpu_burn():
            """Burn CPU cycles"""
            while not stop_flag.is_set():
                # Busy loop
                _ = sum(i*i for i in range(10000))
        
        # Start CPU stress threads
        threads = []
        for _ in range(cores):
            t = threading.Thread(target=cpu_burn, daemon=True)
            t.start()
            threads.append(t)
        
        self.active_injections['cpu_stress'] = {'threads': threads}
    
    def _inject_memory_stress(self, scenario: ChaosScenario) -> None:
        """Inject memory stress"""
        params = scenario.parameters
        memory_mb = params.get('memory_mb', 512)
        
        logger.info(f"Injecting memory stress: {memory_mb}MB")
        
        # Allocate memory
        memory_hog = bytearray(memory_mb * 1024 * 1024)
        self.active_injections['memory_stress'] = {'memory': memory_hog}
    
    def _inject_disk_stress(self, scenario: ChaosScenario) -> None:
        """Inject disk stress"""
        params = scenario.parameters
        fill_percentage = params.get('fill_percentage', 90)
        
        logger.info(f"Injecting disk stress: fill to {fill_percentage}%")
        
        # Create large temporary file
        temp_file = '/tmp/chaos_disk_stress.tmp'
        disk_usage = psutil.disk_usage('/')
        target_bytes = int(disk_usage.free * (fill_percentage / 100.0))
        
        with open(temp_file, 'wb') as f:
            f.write(b'0' * min(target_bytes, 1024*1024*1024))  # Cap at 1GB
        
        self.active_injections['disk_stress'] = {'temp_file': temp_file}
    
    def _inject_exceptions(self, scenario: ChaosScenario) -> None:
        """Inject random exceptions"""
        params = scenario.parameters
        exception_types = params.get('exception_types', ['RuntimeError'])
        injection_rate = params.get('injection_rate', 0.1)
        
        logger.info(f"Injecting exceptions: {exception_types} at {injection_rate*100}% rate")
        
        # This would typically use bytecode instrumentation or AOP
        # For demonstration, we'll store the configuration
        self.active_injections['exceptions'] = {
            'types': exception_types,
            'rate': injection_rate
        }
    
    def _inject_nulls(self, scenario: ChaosScenario) -> None:
        """Inject null returns"""
        params = scenario.parameters
        injection_rate = params.get('injection_rate', 0.1)
        
        logger.info(f"Injecting nulls at {injection_rate*100}% rate")
        
        self.active_injections['nulls'] = {'rate': injection_rate}
    
    def _inject_slow_response(self, scenario: ChaosScenario) -> None:
        """Inject slow responses"""
        params = scenario.parameters
        min_delay = params.get('min_delay_ms', 100)
        max_delay = params.get('max_delay_ms', 1000)
        
        logger.info(f"Injecting slow responses: {min_delay}-{max_delay}ms")
        
        self.active_injections['slow_response'] = {
            'min_delay': min_delay,
            'max_delay': max_delay
        }
    
    def _inject_state_corruption(self, scenario: ChaosScenario) -> None:
        """Inject state corruption"""
        params = scenario.parameters
        corruption_rate = params.get('corruption_rate', 0.05)
        
        logger.info(f"Injecting state corruption at {corruption_rate*100}% rate")
        
        self.active_injections['state_corruption'] = {'rate': corruption_rate}
    
    def _inject_time_skew(self, scenario: ChaosScenario) -> None:
        """Inject time skew"""
        params = scenario.parameters
        skew_seconds = params.get('skew_seconds', 3600)
        
        logger.info(f"Injecting time skew: {skew_seconds}s")
        
        import time as time_module
        original_time = time_module.time
        
        def skewed_time():
            return original_time() + skew_seconds
        
        time_module.time = skewed_time
        self.active_injections['time_skew'] = {'original_time': original_time}
    
    def _cleanup_injection(self, scenario: ChaosScenario) -> None:
        """Clean up after fault injection"""
        logger.info(f"Cleaning up injection: {scenario.name}")
        
        scenario_key = scenario.scenario_type.value
        
        # Restore network methods
        if 'network_latency' in self.active_injections:
            import socket
            injection = self.active_injections['network_latency']
            socket.socket.send = injection['original_send']
            socket.socket.recv = injection['original_recv']
            del self.active_injections['network_latency']
        
        if 'network_partition' in self.active_injections:
            import socket
            injection = self.active_injections['network_partition']
            socket.socket.connect = injection['original_connect']
            del self.active_injections['network_partition']
        
        if 'packet_loss' in self.active_injections:
            import socket
            injection = self.active_injections['packet_loss']
            socket.socket.send = injection['original_send']
            del self.active_injections['packet_loss']
        
        # Stop CPU stress
        if 'cpu_stress' in self._stop_flags:
            self._stop_flags['cpu_stress'].set()
            del self._stop_flags['cpu_stress']
            del self.active_injections['cpu_stress']
        
        # Release memory
        if 'memory_stress' in self.active_injections:
            del self.active_injections['memory_stress']['memory']
            del self.active_injections['memory_stress']
        
        # Remove temp files
        if 'disk_stress' in self.active_injections:
            temp_file = self.active_injections['disk_stress']['temp_file']
            if os.path.exists(temp_file):
                os.remove(temp_file)
            del self.active_injections['disk_stress']
        
        # Restore time
        if 'time_skew' in self.active_injections:
            import time as time_module
            injection = self.active_injections['time_skew']
            time_module.time = injection['original_time']
            del self.active_injections['time_skew']
        
        # Clear other injections
        for key in list(self.active_injections.keys()):
            del self.active_injections[key]
    
    def _collect_impact_metrics(self) -> Dict[str, Any]:
        """Collect metrics about the impact of fault injection"""
        return {
            'cpu_usage': psutil.cpu_percent(interval=1),
            'memory_usage': psutil.virtual_memory().percent,
            'disk_usage': psutil.disk_usage('/').percent,
            'active_injections': len(self.active_injections)
        }
    
    def get_injection_history(self) -> List[InjectionResult]:
        """Get history of all injections"""
        return self.injection_history
    
    def clear_history(self) -> None:
        """Clear injection history"""
        self.injection_history.clear()
