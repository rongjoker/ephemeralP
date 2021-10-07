from typing import List


class Solution:
    def numMatchingSubseq(self, s: str, words: List[str]) -> int:
        ans = 0
        dictionary = [[] for _ in range(26)]
        for w in words:
            it = iter(w)
            dictionary[ord(next(it)) - ord('a')].append(it)
        for le in s:
            cur = ord(le) - ord('a')
            old = dictionary[cur]
            dictionary[cur] = []
            for it in old:
                nex = next(it, None)
                if nex:
                    dictionary[ord(nex) - ord('a')].append(it)
                else:
                    ans += 1
        return ans


a = Solution()

print(a.numMatchingSubseq("abcde", ["a", "bb", "acd", "ace"]))
