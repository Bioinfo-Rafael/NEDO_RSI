from common import Adapter,ROOT
class SearchAgents(Adapter):
    source='searchagents'
    def discover(self):return sorted((ROOT/'raw/search_agents').glob('**/render_*.html'))
    def bundles(self):
        yield from ()
