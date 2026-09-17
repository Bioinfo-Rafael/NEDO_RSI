from adapters.openevolve import OpenEvolve
from adapters.evo_mcts import EvoMCTS
from adapters.rest_mcts import RestMCTS
from adapters.swe_agent import SWEAgent
from adapters.search_agents import SearchAgents
from adapters.dream_rsi import DreamRSI
from adapters.treeofthoughts import TreeOfThoughts
ADAPTERS={c.source:c for c in (OpenEvolve,EvoMCTS,RestMCTS,SWEAgent,SearchAgents,DreamRSI,TreeOfThoughts)}
