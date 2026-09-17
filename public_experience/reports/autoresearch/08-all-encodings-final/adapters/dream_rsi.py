from common import Adapter
class DreamRSI(Adapter):
    source='dreamrsi'
    def bundles(self):
        # Pinned repository has a paper and release plan, no execution records.
        yield from ()
