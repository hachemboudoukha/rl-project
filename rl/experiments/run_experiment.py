import time
from ..utils.logger import setup_logger
from ..utils.serialization import save_object

def run_experiment(agent, episodes, name="experiment", save_path=None):
    logger = setup_logger(name, f"logs/{name}.log")
    logger.info(f"Starting experiment: {name}")
    
    start_time = time.time()
    policy, Q = agent.train(episodes)
    end_time = time.time()
    
    duration = end_time - start_time
    logger.info(f"Experiment finished in {duration:.2f}s")
    
    if save_path:
        save_object({'policy': policy, 'Q': Q}, save_path)
        logger.info(f"Results saved to {save_path}")
    
    return policy, Q
