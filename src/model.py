import json
import sys
import Levenshtein

def clear(s):
    return s.replace('\n','').replace(' ','').replace(' ','')

def longest_common_subsequence(nums1, nums2):
    m, n = len(nums1), len(nums2)
    if m==0 or n==0:
        return 0
    dp = [[0] * (n+1) for _ in range(m+1)]
    for i in range(1, m+1 ):
        for j in range(1, n+1):
            if nums1[i -1] == nums2[j-1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[m][n]

def compare(s1,s2):
    siz=min(len(s1),len(s2))
    return longest_common_subsequence(s1,s2)>siz-10

print('ga')

file=sys.argv[1]
with open(file+'\\source.json','r') as f:
    text=json.load(f)

with open(file+'\\newdata\\code-edit.json','r',encoding='utf-8') as f:
    data=json.load(f)

for i in range(len(data)):
    if compare(clear(text['code']),clear(data[str(i)]['code'])):
        code=data[str(i)]['code']
        truth=data[str(i)]['new']
        edit={'data':data[str(i)]['edit']}
        with open(file+'\\source.json','w') as f:
            json.dump(edit,f)
        break
