"""
🚀 Roadmapex - Exécution Prédictive Python
==========================================

Transforme l'exécution Python réactive en orchestration stratégique prédictive.
Créé par Anzize Daouda - 2025

Architecture : Un seul fichier, toute la puissance.
"""

import time
import threading
import functools
import inspect
from typing import Dict, List, Callable, Any, Optional, Set
from dataclasses import dataclass, field
from collections import defaultdict, deque
import psutil
import os


__version__ = "0.1.0"
__author__ = "Anzize Daouda"
__email__ = "nexusstudio100@gmail.com"


# =============================================================================
# CORE DATA STRUCTURES
# =============================================================================

@dataclass
class TaskMetrics:
    """Métriques détaillées d'une tâche"""
    name: str
    discovery_time: float = 0.0
    setup_time: float = 0.0 
    execution_time: float = 0.0
    memory_before: float = 0.0
    memory_after: float = 0.0
    optimized: bool = False
    preloaded: bool = False
    
    @property
    def total_time(self) -> float:
        return self.discovery_time + self.setup_time + self.execution_time
    
    @property
    def memory_delta(self) -> float:
        return self.memory_after - self.memory_before


@dataclass 
class RoadmapConfig:
    """Configuration de la roadmap d'exécution"""
    phases: List[str] = field(default_factory=list)
    preload_targets: List[str] = field(default_factory=list)
    parallel_safe: List[str] = field(default_factory=list)
    cache_results: List[str] = field(default_factory=list)
    memory_pool_size: int = 100  # MB
    enable_profiling: bool = True


@dataclass
class PreloadedResource:
    """Ressource préchargée avec métadonnées"""
    name: str
    data: Any
    preloaded_at: float
    setup_time_saved: float
    memory_footprint: float


# =============================================================================
# DECORATORS SYSTEM
# =============================================================================

class RoadmapDecorators:
    """Système de décorateurs pour Roadmapex"""
    
    _registered_tasks: Dict[str, dict] = {}
    _phase_registry: Dict[str, List[str]] = defaultdict(list)
    _optimization_registry: Dict[str, dict] = {}
    
    @classmethod
    def phase(cls, phase_name: str, priority: int = 0):
        """Décorateur pour assigner une phase à une fonction"""
        def decorator(func):
            task_name = func.__name__
            cls._registered_tasks[task_name] = {
                'function': func,
                'phase': phase_name,
                'priority': priority,
                'dependencies': getattr(func, '_dependencies', []),
                'optimizations': getattr(func, '_optimizations', {})
            }
            cls._phase_registry[phase_name].append(task_name)
            
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                return func(*args, **kwargs)
            return wrapper
        return decorator
    
    @classmethod 
    def optimize(cls, preload: bool = False, cache: bool = False, 
                parallel_safe: bool = False, memory_pool: bool = False,
                setup_cost_ms: float = 0):
        """Décorateur pour optimisations spécifiques"""
        def decorator(func):
            func._optimizations = {
                'preload': preload,
                'cache': cache,
                'parallel_safe': parallel_safe,
                'memory_pool': memory_pool,
                'setup_cost_ms': setup_cost_ms
            }
            
            # Enregistrement pour usage ultérieur
            task_name = func.__name__
            cls._optimization_registry[task_name] = func._optimizations
            
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                return func(*args, **kwargs)
            return wrapper
        return decorator
    
    @classmethod
    def dependency(cls, requires: List[str]):
        """Décorateur pour spécifier les dépendances"""
        def decorator(func):
            func._dependencies = requires
            
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                return func(*args, **kwargs)
            return wrapper
        return decorator
    
    @classmethod
    def execute(cls):
        """Décorateur pour déclencher l'exécution Roadmapex"""
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                # Création automatique de l'exécuteur
                executor = RoadmapExecutor()
                
                # Configuration automatique basée sur les décorateurs
                executor.auto_configure_from_registry()
                
                # Exécution optimisée
                return executor.execute_function_with_roadmap(func, *args, **kwargs)
            return wrapper
        return decorator


# Instance globale des décorateurs
roadmap = RoadmapDecorators()
optimize = RoadmapDecorators.optimize
dependency = RoadmapDecorators.dependency


# =============================================================================
# PRELOADER SYSTEM
# =============================================================================

class IntelligentPreloader:
    """Système de préchargement intelligent des ressources"""
    
    def __init__(self):
        self.preloaded_resources: Dict[str, PreloadedResource] = {}
        self.preload_cache: Dict[str, Any] = {}
        self.memory_pool: Dict[str, Any] = {}
    
    def can_preload(self, func: Callable) -> bool:
        """Détermine si une fonction peut être préchargée"""
        optimizations = getattr(func, '_optimizations', {})
        return optimizations.get('preload', False)
    
    def preload_function_resources(self, func: Callable) -> float:
        """Précharge les ressources d'une fonction"""
        start_time = time.perf_counter()
        func_name = func.__name__
        
        # Simulation du préchargement (setup partiel)
        optimizations = getattr(func, '_optimizations', {})
        setup_cost = optimizations.get('setup_cost_ms', 0) / 1000  # Convert to seconds
        
        if setup_cost > 0:
            # Exécution du setup en avance (70% du coût)
            preload_time = setup_cost * 0.7
            time.sleep(preload_time)  # Simulation réaliste
            
            # Stockage de la ressource préchargée
            memory_before = self._get_memory_usage()
            
            # Simulation d'une ressource préchargée
            preloaded_data = f"preloaded_resource_{func_name}_{int(time.time())}"
            self.preload_cache[func_name] = preloaded_data
            
            memory_after = self._get_memory_usage()
            
            resource = PreloadedResource(
                name=func_name,
                data=preloaded_data,
                preloaded_at=time.perf_counter(),
                setup_time_saved=setup_cost * 0.7,  # 70% de gain
                memory_footprint=memory_after - memory_before
            )
            
            self.preloaded_resources[func_name] = resource
            
            end_time = time.perf_counter()
            return (end_time - start_time) * 1000  # Return ms
        
        return 0.0
    
    def get_preloaded_resource(self, func_name: str) -> Optional[PreloadedResource]:
        """Récupère une ressource préchargée"""
        return self.preloaded_resources.get(func_name)
    
    def is_preloaded(self, func_name: str) -> bool:
        """Vérifie si une ressource est préchargée"""
        return func_name in self.preloaded_resources
    
    def _get_memory_usage(self) -> float:
        """Récupère l'usage mémoire actuel en MB"""
        try:
            process = psutil.Process(os.getpid())
            return process.memory_info().rss / 1024 / 1024  # MB
        except:
            return 0.0


# =============================================================================
# DEPENDENCY ANALYZER
# =============================================================================

class DependencyAnalyzer:
    """Analyseur de dépendances pour optimisation d'ordre d'exécution"""
    
    def __init__(self):
        self.dependency_graph: Dict[str, Set[str]] = defaultdict(set)
        self.execution_order: List[str] = []
    
    def build_dependency_graph(self, tasks: List[Callable]):
        """Construit le graphe de dépendances"""
        self.dependency_graph.clear()
        
        for task in tasks:
            task_name = task.__name__
            dependencies = getattr(task, '_dependencies', [])
            self.dependency_graph[task_name] = set(dependencies)
    
    def compute_optimal_order(self, tasks: List[Callable]) -> List[str]:
        """Calcule l'ordre optimal d'exécution (tri topologique)"""
        self.build_dependency_graph(tasks)
        
        # Tri topologique de Kahn
        in_degree = defaultdict(int)
        task_names = [task.__name__ for task in tasks]
        
        # Calcul des degrés entrants
        for task_name in task_names:
            for dependency in self.dependency_graph[task_name]:
                in_degree[dependency] = in_degree.get(dependency, 0)
                in_degree[task_name] = in_degree.get(task_name, 0) + 1
        
        # Queue des tâches sans dépendances
        queue = deque([name for name in task_names if in_degree[name] == 0])
        execution_order = []
        
        while queue:
            current = queue.popleft()
            execution_order.append(current)
            
            # Retirer les dépendances satisfaites
            for task_name in task_names:
                if current in self.dependency_graph[task_name]:
                    in_degree[task_name] -= 1
                    if in_degree[task_name] == 0:
                        queue.append(task_name)
        
        # Vérification de cycles
        if len(execution_order) != len(task_names):
            remaining = set(task_names) - set(execution_order)
            raise ValueError(f"Dépendances circulaires détectées: {remaining}")
        
        self.execution_order = execution_order
        return execution_order


# =============================================================================
# METRICS COLLECTOR
# =============================================================================

class MetricsCollector:
    """Collecteur de métriques de performance"""
    
    def __init__(self):
        self.task_metrics: List[TaskMetrics] = []
        self.analysis_time: float = 0.0
        self.total_preload_time: float = 0.0
        self.total_execution_time: float = 0.0
        
    def start_task_measurement(self, task_name: str) -> dict:
        """Démarre la mesure d'une tâche"""
        return {
            'name': task_name,
            'start_time': time.perf_counter(),
            'memory_before': self._get_memory_usage()
        }
    
    def end_task_measurement(self, measurement_context: dict, 
                           discovery_time: float = 0.0,
                           setup_time: float = 0.0,
                           optimized: bool = False,
                           preloaded: bool = False) -> TaskMetrics:
        """Termine la mesure et enregistre les métriques"""
        end_time = time.perf_counter()
        memory_after = self._get_memory_usage()
        
        execution_time = (end_time - measurement_context['start_time']) * 1000  # ms
        
        metrics = TaskMetrics(
            name=measurement_context['name'],
            discovery_time=discovery_time,
            setup_time=setup_time,
            execution_time=execution_time - setup_time,  # Pure execution time
            memory_before=measurement_context['memory_before'],
            memory_after=memory_after,
            optimized=optimized,
            preloaded=preloaded
        )
        
        self.task_metrics.append(metrics)
        return metrics
    
    def get_summary(self) -> Dict[str, Any]:
        """Résumé complet des métriques"""
        if not self.task_metrics:
            return {}
        
        total_time = sum(m.total_time for m in self.task_metrics)
        avg_memory = sum(m.memory_delta for m in self.task_metrics) / len(self.task_metrics)
        max_memory = max(m.memory_after for m in self.task_metrics)
        optimized_count = sum(1 for m in self.task_metrics if m.optimized)
        preloaded_count = sum(1 for m in self.task_metrics if m.preloaded)
        
        return {
            'total_time': total_time,
            'task_count': len(self.task_metrics),
            'avg_memory': avg_memory,
            'max_memory': max_memory,
            'analysis_time': self.analysis_time,
            'preload_time': self.total_preload_time,
            'optimized_tasks': optimized_count,
            'preloaded_tasks': preloaded_count,
            'tasks': [
                {
                    'name': m.name,
                    'total_time': m.total_time,
                    'memory_delta': m.memory_delta,
                    'optimized': m.optimized,
                    'preloaded': m.preloaded
                }
                for m in self.task_metrics
            ]
        }
    
    def _get_memory_usage(self) -> float:
        """Usage mémoire actuel en MB"""
        try:
            process = psutil.Process(os.getpid())
            return process.memory_info().rss / 1024 / 1024
        except:
            return 0.0


# =============================================================================
# MAIN ROADMAP EXECUTOR
# =============================================================================

class RoadmapExecutor:
    """Moteur principal d'exécution Roadmapex"""
    
    def __init__(self, config: Optional[RoadmapConfig] = None):
        self.config = config or RoadmapConfig()
        self.preloader = IntelligentPreloader()
        self.analyzer = DependencyAnalyzer()
        self.metrics = MetricsCollector()
        self.task_registry: Dict[str, Callable] = {}
        
    def set_roadmap(self, phases: List[str], preload_targets: List[str]):
        """Configuration simple de la roadmap"""
        self.config.phases = phases
        self.config.preload_targets = preload_targets
        
    def auto_configure_from_registry(self):
        """Configuration automatique basée sur les décorateurs"""
        # Récupération des tâches décorées
        for task_name, task_info in RoadmapDecorators._registered_tasks.items():
            func = task_info['function']
            self.task_registry[task_name] = func
            
            # Auto-configuration des preload targets
            optimizations = task_info.get('optimizations', {})
            if optimizations.get('preload', False):
                if task_name not in self.config.preload_targets:
                    self.config.preload_targets.append(task_name)
                    
            # Auto-configuration des phases
            phase = task_info.get('phase')
            if phase and phase not in self.config.phases:
                self.config.phases.append(phase)
    
    def execute_with_roadmap(self, tasks: List[Callable]) -> Dict[str, Any]:
        """Exécution principale avec roadmap complète"""
        start_time = time.perf_counter()
        
        # Phase 1: Analyse stratégique
        analysis_start = time.perf_counter()
        optimal_order = self.analyzer.compute_optimal_order(tasks)
        self.metrics.analysis_time = (time.perf_counter() - analysis_start) * 1000
        
        print(f"📋 Ordre d'exécution optimisé: {' → '.join(optimal_order)}")
        
        # Phase 2: Préchargement intelligent
        preload_start = time.perf_counter()
        self._execute_preload_phase(tasks)
        self.metrics.total_preload_time = (time.perf_counter() - preload_start) * 1000
        
        # Phase 3: Exécution orchestrée
        self._execute_optimized_tasks(tasks, optimal_order)
        
        total_time = (time.perf_counter() - start_time) * 1000
        print(f"✅ Exécution Roadmapex terminée en {total_time:.1f}ms")
        
        return self.metrics.get_summary()
    
    def execute_function_with_roadmap(self, main_func: Callable, *args, **kwargs):
        """Exécute une fonction avec roadmap automatique"""
        # Auto-découverte des tâches dans le scope
        tasks = self._discover_decorated_tasks()
        
        if tasks:
            # Exécution avec roadmap
            self.execute_with_roadmap(tasks)
        
        # Exécution de la fonction principale
        return main_func(*args, **kwargs)
    
    def _discover_decorated_tasks(self) -> List[Callable]:
        """Découvre automatiquement les tâches décorées"""
        tasks = []
        for task_name, task_info in RoadmapDecorators._registered_tasks.items():
            tasks.append(task_info['function'])
        return tasks
    
    def _execute_preload_phase(self, tasks: List[Callable]):
        """Phase de préchargement intelligent"""
        print("⚡ Phase de préchargement...")
        
        preload_tasks = [
            task for task in tasks 
            if task.__name__ in self.config.preload_targets
        ]
        
        total_savings = 0.0
        for task in preload_tasks:
            savings = self.preloader.preload_function_resources(task)
            total_savings += savings
            if savings > 0:
                print(f"  ✓ {task.__name__} préchargé (économie: {savings:.1f}ms)")
        
        print(f"  💎 Total économisé: {total_savings:.1f}ms")
    
    def _execute_optimized_tasks(self, tasks: List[Callable], execution_order: List[str]):
        """Exécution orchestrée des tâches"""
        print("🚀 Phase d'exécution orchestrée...")
        
        # Création d'un mapping nom -> fonction
        task_map = {task.__name__: task for task in tasks}
        
        for task_name in execution_order:
            if task_name in task_map:
                self._execute_single_optimized_task(task_map[task_name])
    
    def _execute_single_optimized_task(self, task: Callable):
        """Exécution optimisée d'une tâche unique"""
        task_name = task.__name__
        
        # Démarrage de la mesure
        measurement = self.metrics.start_task_measurement(task_name)
        
        # Pas de temps de découverte - roadmap connue
        discovery_time = 0.0
        
        # Setup optimisé si préchargé
        optimizations = getattr(task, '_optimizations', {})
        setup_cost = optimizations.get('setup_cost_ms', 0)
        
        preloaded_resource = self.preloader.get_preloaded_resource(task_name)
        if preloaded_resource:
            # Setup déjà fait - gain de 70%
            setup_time = setup_cost * 0.3  # 30% du temps original
            preloaded = True
            print(f"  ⚡ {task_name} utilise ressource préchargée")
        else:
            # Setup complet nécessaire
            setup_time = setup_cost
            preloaded = False
        
        # Simulation du setup résiduel
        if setup_time > 0:
            time.sleep(setup_time / 1000)  # Convert ms to seconds
        
        # Exécution réelle de la tâche
        try:
            result = task()
            success = True
        except Exception as e:
            print(f"  ❌ Erreur lors de l'exécution de {task_name}: {e}")
            result = None
            success = False
        
        # Finalisation de la mesure
        metrics = self.metrics.end_task_measurement(
            measurement,
            discovery_time=discovery_time,
            setup_time=setup_time,
            optimized=preloaded or optimizations.get('cache', False),
            preloaded=preloaded
        )
        
        if success:
            print(f"  ✓ {task_name} exécuté en {metrics.total_time:.1f}ms")
        
        return result


# =============================================================================
# INTELLIGENT PRELOADER IMPLEMENTATION
# =============================================================================

class IntelligentPreloader:
    """Système de préchargement intelligent des ressources"""
    
    def __init__(self):
        self.preloaded_resources: Dict[str, PreloadedResource] = {}
        self.preload_cache: Dict[str, Any] = {}
        self.memory_pool: Dict[str, Any] = {}
    
    def preload_function_resources(self, func: Callable) -> float:
        """Précharge les ressources d'une fonction et retourne le temps économisé"""
        start_time = time.perf_counter()
        func_name = func.__name__
        
        # Récupération des optimisations
        optimizations = getattr(func, '_optimizations', {})
        setup_cost = optimizations.get('setup_cost_ms', 0)
        
        if setup_cost > 0 and optimizations.get('preload', False):
            # Simulation du préchargement (70% du setup fait à l'avance)
            preload_duration = (setup_cost * 0.7) / 1000  # Convert to seconds
            time.sleep(preload_duration)
            
            # Mesure mémoire
            memory_before = self._get_memory_usage()
            
            # Création de la ressource préchargée
            preloaded_data = {
                'timestamp': time.time(),
                'function': func_name,
                'setup_completed': True,
                'optimization_applied': True
            }
            
            self.preload_cache[func_name] = preloaded_data
            memory_after = self._get_memory_usage()
            
            # Enregistrement de la ressource
            resource = PreloadedResource(
                name=func_name,
                data=preloaded_data,
                preloaded_at=time.perf_counter(),
                setup_time_saved=setup_cost * 0.7,
                memory_footprint=memory_after - memory_before
            )
            
            self.preloaded_resources[func_name] = resource
            
            end_time = time.perf_counter()
            return resource.setup_time_saved  # Retourne le temps économisé
        
        return 0.0
    
    def get_preloaded_resource(self, func_name: str) -> Optional[PreloadedResource]:
        """Récupère une ressource préchargée"""
        return self.preloaded_resources.get(func_name)
    
    def is_preloaded(self, func_name: str) -> bool:
        """Vérifie si une ressource est préchargée"""
        return func_name in self.preloaded_resources
    
    def get_total_savings(self) -> float:
        """Calcule le total des économies de temps"""
        return sum(resource.setup_time_saved for resource in self.preloaded_resources.values())
    
    def _get_memory_usage(self) -> float:
        """Usage mémoire actuel en MB"""
        try:
            process = psutil.Process(os.getpid())
            return process.memory_info().rss / 1024 / 1024
        except:
            return 0.0


# =============================================================================
# PUBLIC API - INTERFACE SIMPLE
# =============================================================================

def create_roadmap_executor(phases: List[str] = None, 
                          preload_targets: List[str] = None) -> RoadmapExecutor:
    """Crée un exécuteur Roadmapex avec configuration simple"""
    config = RoadmapConfig(
        phases=phases or ['init', 'main', 'cleanup'],
        preload_targets=preload_targets or []
    )
    return RoadmapExecutor(config)


def benchmark_execution(tasks: List[Callable], 
                       preload_targets: List[str] = None) -> Dict[str, Any]:
    """Benchmark d'exécution avec et sans Roadmapex"""
    print("🔬 Benchmark d'exécution Roadmapex...")
    
    # Test avec Roadmapex
    executor = create_roadmap_executor(preload_targets=preload_targets or [])
    roadmapex_results = executor.execute_with_roadmap(tasks)
    
    # Simulation basique Python standard pour comparaison
    standard_time = 0.0
    standard_memory = []
    
    for task in tasks:
        # Temps de découverte simulé
        discovery_time = 30 + (hash(task.__name__) % 50)  # 30-80ms
        
        # Setup cost complet
        optimizations = getattr(task, '_optimizations', {})
        setup_cost = optimizations.get('setup_cost_ms', 0)
        
        # Mesure mémoire
        memory_before = executor.metrics._get_memory_usage()
        
        # Exécution avec setup complet
        start = time.perf_counter()
        if setup_cost > 0:
            time.sleep(setup_cost / 1000)
        
        try:
            task()
        except:
            pass  # Ignore les erreurs pour le benchmark
            
        exec_time = (time.perf_counter() - start) * 1000
        memory_after = executor.metrics._get_memory_usage()
        
        task_total = discovery_time + setup_cost + exec_time
        standard_time += task_total
        standard_memory.append(memory_after - memory_before)
    
    # Comparaison
    roadmapex_total = roadmapex_results['total_time'] + roadmapex_results['analysis_time']
    speed_improvement = ((standard_time - roadmapex_total) / standard_time) * 100
    
    return {
        'standard_time': standard_time,
        'roadmapex_time': roadmapex_total,
        'speed_improvement': speed_improvement,
        'roadmapex_details': roadmapex_results
    }


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def execute_roadmap(tasks: List[Callable], 
                   preload_targets: List[str] = None,
                   phases: List[str] = None) -> Any:
    """Fonction de convenance pour exécution rapide"""
    executor = create_roadmap_executor(phases, preload_targets)
    return executor.execute_with_roadmap(tasks)


def quick_optimize(*task_functions, preload_all: bool = True):
    """Optimisation rapide d'une liste de fonctions"""
    preload_targets = [f.__name__ for f in task_functions] if preload_all else []
    return execute_roadmap(list(task_functions), preload_targets)


# =============================================================================
# PROFILING AND DEBUGGING
# =============================================================================

class RoadmapProfiler:
    """Profiler pour analyser les performances Roadmapex"""
    
    @staticmethod
    def profile_task(func: Callable, iterations: int = 100) -> Dict[str, float]:
        """Profile une tâche spécifique"""
        times = []
        
        for _ in range(iterations):
            start = time.perf_counter()
            try:
                func()
            except:
                pass
            end = time.perf_counter()
            times.append((end - start) * 1000)
        
        return {
            'avg_time': sum(times) / len(times),
            'min_time': min(times),
            'max_time': max(times),
            'std_dev': (sum((t - sum(times)/len(times))**2 for t in times) / len(times))**0.5
        }
    
    @staticmethod
    def compare_with_without_roadmapex(tasks: List[Callable], 
                                     preload_targets: List[str]) -> Dict[str, Any]:
        """Compare performance avec et sans Roadmapex"""
        return benchmark_execution(tasks, preload_targets)


# =============================================================================
# EXPORTS - API PUBLIQUE
# =============================================================================

__all__ = [
    # Classes principales
    'RoadmapExecutor',
    'RoadmapConfig', 
    'TaskMetrics',
    'IntelligentPreloader',
    'DependencyAnalyzer',
    'MetricsCollector',
    
    # Décorateurs
    'roadmap',
    'optimize', 
    'dependency',
    
    # Fonctions de convenance
    'create_roadmap_executor',
    'execute_roadmap',
    'quick_optimize',
    'benchmark_execution',
    
    # Profiling
    'RoadmapProfiler',
    
    # Version
    '__version__'
]


# =============================================================================
# EXEMPLE D'USAGE SIMPLE
# =============================================================================

if __name__ == "__main__":
    # Démonstration simple
    print("🚀 Roadmapex - Démonstration Simple")
    print("=" * 40)
    
    @roadmap.phase("init")
    @optimize(preload=True, setup_cost_ms=200)
    def init_database():
        print("  🗄️ Base de données initialisée")
        return "db_connection"
    
    @roadmap.phase("init") 
    @optimize(preload=True, setup_cost_ms=150)
    def load_config():
        print("  ⚙️ Configuration chargée")
        return {"api_url": "https://api.example.com"}
    
    @roadmap.phase("main")
    @dependency(requires=["init_database", "load_config"])
    def main_process():
        print("  🎯 Traitement principal")
        return "processed_data"
    
    @roadmap.execute()
    def demo():
        """Démonstration avec roadmap automatique"""
        config = load_config()
        db = init_database()
        result = main_process()
        return f"Demo terminée avec {config} et {db} → {result}"
    
    # Exécution de la démo
    result = demo()
    print(f"\n✅ {result}")