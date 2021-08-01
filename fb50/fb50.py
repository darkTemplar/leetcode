

from collections import defaultdict

def isAlienSorted(words: List[str], order: str) -> bool:
    orderMap = convertToMap(order)




def convertToMap(order: List[str]) ->  Dict[str, int]:
    wordMap = defaultdict(int)
    for i, s in enumerate(order):
        wordMap[s] = i
    return wordMap
