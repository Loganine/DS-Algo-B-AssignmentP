# Assignment #1: 拉齐大家Python水平

Updated 0940 GMT+8 Feb 19, 2024

2024 spring, Complied by ==焦晨航 数学科学学院==



**说明：**

1）数算课程的先修课是计概，由于计概学习中可能使用了不同的编程语言，而数算课程要求Python语言，因此第一周作业练习Python编程。如果有同学坚持使用C/C++，也可以，但是建议也要会Python语言。

2）请把每个题目解题思路（可选），源码Python, 或者C++（已经在Codeforces/Openjudge上AC），截图（包含Accepted），填写到下面作业模版中（推荐使用 typora https://typoraio.cn ，或者用word）。AC 或者没有AC，都请标上每个题目大致花费时间。

3）课程网站是Canvas平台, https://pku.instructure.com, 学校通知3月1日导入选课名单后启用。**作业写好后，保留在自己手中，待3月1日提交。**

提交时候先提交pdf文件，再把md或者doc文件上传到右侧“作业评论”。Canvas需要有同学清晰头像、提交文件有pdf、"作业评论"区有上传的md或者doc附件。

4）如果不能在截止前提交作业，请写明原因。



**编程环境**

==（请改为同学的操作系统、编程环境等）==

操作系统：Windows 11 家庭中文版64位（10.0，版本22621）

Python编程环境：Basthon Python 3编程环境

C/C++编程环境：Mac terminal vi (version 9.0.1424), g++/gcc (Apple clang version 14.0.3, clang-1403.0.22.14.1)(我不会这一语言)



## 1. 题目

### 20742: 泰波拿契數

http://cs101.openjudge.cn/practice/20742/



思路：直接打一个以输入数字为上限的列表，输出列表的最后一位。



##### 代码
n=int(input())
lis=[]
i=0
while i<=n:
    if i==0:
        lis.append(0)
    elif i==1 or i==2:
        lis.append(1)
    else:
        lis.append(lis[i-2]+lis[i-1]+lis[i-3])
    i+=1
print(lis[-1])
```python
# 

```



代码运行截图 ==（至少包含有"Accepted"）==





### 58A. Chat room

greedy/strings, 1000, http://codeforces.com/problemset/problem/58/A



思路：创建一个与‘hello’字母一一对应的数组，然后挨个查找字母



##### 代码

```python
# 
s=input().lower()
dp=[0]*5
ans='hello'
num=0
for i in s:
    if i==ans[num]:
        dp[num]+=1
        num+=1
    if num==5:
        break
if sum(dp)==5:
    print('YES')
else:
    print('NO')
```



代码运行截图 ==（至少包含有"Accepted"）==





### 118A. String Task

implementation/strings, 1000, http://codeforces.com/problemset/problem/118/A



思路：感觉按题目走就行……没有什么值得一提的思路，连我都能一遍过



##### 代码
st=input().lower()
ans=''
lis=['a','e','i','o','u','y']
for i in st:
    if i not in lis:
        ans+='.'
        ans+=i
print(ans)
```python
# 

```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==





### 22359: Goldbach Conjecture

http://cs101.openjudge.cn/practice/22359/



思路：写了一个经典的判断质数的函数



##### 代码
n=int(input())
def isprime(n):
    for i in range(2,int(n**0.5)+1):
        if n%i==0:
            return False
    return True
for i in range(2,n//2):
    if isprime(i) and isprime(n-i):
        print(i,n-i)
        break
```python
# 

```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==





### 23563: 多项式时间复杂度

http://cs101.openjudge.cn/practice/23563/



思路：关键是找到每一项的次数。这里借鉴使用了split函数将‘+’‘n^’作为分割标志的功能。



##### 代码
st=input().split('+')
new=[i.split('n^') for i in st]
ans=0
for i in new:
    if int(i[1])>ans and i[0]!='0':
        ans=int(i[1])
print('n^%d'%(ans))
```python
# 

```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==





### 24684: 直播计票

http://cs101.openjudge.cn/practice/24684/



思路：我以前不太会用字典，只好借鉴：创建一个去重的字典（用到defaultdict），统计票数，将得票最多者放入新列表，排序输出



##### 代码
from collections import defaultdict
votes = list(map(int, input().split()))
vote_counts = defaultdict(int)
for vote in votes:
    vote_counts[vote] += 1
max_votes = max(vote_counts.values())
winners = sorted([item for item in vote_counts.items() if item[1] == max_votes])
print(' '.join(str(winner[0]) for winner in winners))
```python
# 

```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==





## 2. 学习总结和收获
我个人是属于大一之前基本没接触过编程、计概课程只能勉强跟上，当时机考8道题AC6不是没时间，是后两道真不会……现在数算课程刚开始就有一种力不从心的感觉，栈、树、链表的应用几乎是第一次了解。特别是编程的时候每次感觉思路对了调试也成功了却老是WA或者RE，我认为还是对python和计算机语言不够熟悉，习惯用人类的思路去思考。还有个毛病是光会用列表，其他数据类型不熟悉。
不过好在平行班竞争压力没有那么大，大佬也不少，依然有代码敲敲敲、水平涨涨涨的余地。我并不奢求满分，能否优秀也随缘，自己能在这门课程中学到东西才是最重要的。另外，AC的瞬间真的很有成就感！




