from collections import deque, defaultdict
from heapq import heappush, heappop

def maxSlidingWindow(nums: List[int], k: int) -> List[int]:
        dQ, res = deque(), []
        for i, n in enumerate(nums):
            while dQ and n > nums[dQ[-1]]:
                dQ.pop()
            dQ.append(i)
            # remove head if its outside sliding window
            if dQ[0] == i-k:
                dQ.popleft()
            if i >= k-1:
                res.append(nums[dQ[0]])
        return res

#P1058 - Minimize rounding Error
def minimizeError(self, prices: List[str], target: int) -> str:
        low, high, res = 0, 0, 0
        pq = []
        for p in prices:
            p = float(p)
            low, high = floor(p), ceil(p)
            if low != high:
                priority = (high - p) - (p - low)
                heappush(pq, (priority, p))
            target -= low
            res += p - low
        if target < 0.0 or target > len(pq):
            return "-1"
        while target:
            p, _ = heappop(pq)
            target -= 1
            res += p
        res = "%0.3f" % (res)
        return res


# P787 - Cheapest FLight with K hops at most (using Djikstra's)
def findCheapestPrice(self, n: int, flights: List[List[int]], src: int, dst: int, k: int) -> int:
        W = self.constructFlightGraph(flights)
        #print("Flight Graph ", W)
        Q, D, H, P = [(0, 0, src)], {src: 0}, {src: 0}, {}
        while Q:
            _, stops, u = heappop(Q)
            if u == dst:
                return D.get(u, float('inf'))
            #print("Node ", u)
            if stops > k: 
                #print("Stops exceeded ", H[u])
                continue
            #print("Node bypassed checks ", H)
            for v in W[u]:
                if self.relax(W, u, v, D, P, H):
                    heappush(Q, (D[v], H[v], v))
        return -1
        
      
def relax(self, W, u, v, D, P, H):
    inf = float('inf')
    d = D.get(u, inf) + W[u][v]
    h = H.get(u, inf) + 1
    if d < D.get(v, inf) or h < H.get(v, inf) :
        D[v], P[v], H[v] = d, u, h
        return True
    return False
        
        
        
def constructFlightGraph(self, flights):
    W = defaultdict(dict)
    for f in flights:
        src, dst, cost = f
        if src not in W:
            W[src] = defaultdict(int)
        W[src][dst] = cost
    return W