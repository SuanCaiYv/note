## 力扣所有刷过的hard题

#### [1575. 统计所有可行路径](https://leetcode-cn.com/problems/count-all-possible-routes/)

题目：

给你一个 互不相同 的整数数组，其中 locations[i] 表示第 i 个城市的位置。同时给你 start，finish 和 fuel 分别表示出发城市、目的地城市和你初始拥有的汽油总量

每一步中，如果你在城市 i ，你可以选择任意一个城市 j ，满足  j != i 且 0 <= j < locations.length ，并移动到城市 j 。从城市 i 移动到 j 消耗的汽油量为 |locations[i] - locations[j]|，|x| 表示 x 的绝对值。

请注意， fuel 任何时刻都 不能 为负，且你 可以 经过任意城市超过一次（包括 start 和 finish ）。

请你返回从 start 到 finish 所有可能路径的数目。

由于答案可能很大， 请将它对 10^9 + 7 取余后返回。



解答：

这是一道很简单的DP，直接算肯定算不出来，我们定义dp\[i\]\[j\]表示当剩余油量为i时，以j为起始节点的路径数。

所以题解如下：

``` Java
class Solution {
    public int countRoutes(int[] locations, int start, int finish, int fuel) {
        int mod = 1000000007;
        int len = locations.length;
        // 当剩余油量为m时，到达n点的路径个数
        int[][] dp = new int[210][110];
        dp[0][start] = 1;
        int ans = 0;
        for (int k = 0; k <= fuel; ++ k) {
            for (int i = 0; i < len; ++ i) {
                for (int j = 0; j < len; ++ j) {
                    if (i == j) continue;
                    int distance = Math.abs(locations[i] - locations[j]);
                    if (distance > k) {
                        continue;
                    }
                  	// 这里是动态转移方程l
                    dp[k][i] += dp[k - distance][j];
                    dp[k][i] %= mod;
                }
            }
        }
        for (int k = 0; k <= fuel; ++ k) {
            ans += dp[k][finish];
            ans %= mod;
        }
        return ans;
    }
}
```

#### [992. K 个不同整数的子数组](https://leetcode-cn.com/problems/subarrays-with-k-different-integers/)

题目：

给定一个正整数数组 A，如果 A 的某个子数组中不同整数的个数恰好为 K，则称 A 的这个连续、不一定不同的子数组为好子数组。

（例如，[1,2,3,1,2] 中有 3 个不同的整数：1，2，以及 3。）

返回 A 中好子数组的数目。



解答：

这道题是数学中**恰好转换成最多**的问题。求恰好K个，那我求最多K个f(k)，然后用f(k) - f(k-1)不就行了吗？最多K个很容易用滑动窗口求出来。

``` Java
class Solution {
    
    private int[] rcd;

    private int size = 0;

    public int subarraysWithKDistinct(int[] nums, int k) {
        return f(nums, k) - f(nums, k - 1);
    }

    // subarraysWithKDistinct(k) = f(k) - f(k - 1);
    // f(k)表示整个数组里，最多有k个不同整数的子区间个数。
    // 恰好有k个不同整数的子区间个数=最多有k个不同整数的子区间个数-最多有k-1个不同整数的子区间个数。
    private int f(int[] nums, int k) {
        rcd = new int[nums.length + 1];
        size = 0;
        int l = 0, r = 0;
        int count = 0;
        while (r < nums.length) {
            put(nums[r++]);
            while (size > k) {
                remove(nums[l++]);
            }
            // 相当巧妙啊！这行我看了许久，首先就是对于右边界未触及的情况，所有子区间个数等于r-l的总和
            // 因为在区间[l, r)之间，所有子区间个数=(r-l) + (r-l-1) + (r-l-2) + ... + 1; 恰巧等于循环得到的(r-l)总和(因为r在增加，l不变)。
            // 而对于右边界触及的情况，此时更新l之后，在[l, r]之间，[l, r - 1]已经被计算过了，还剩[l + 1, r], [l + 2, r] ... [r - 1, r]没有计算过
            // 而这恰恰等于(r - l)，所以维护了数量。
            // 官方这么解释：所有的 左边界固定前提下，根据右边界最右的下标，计算出来的子区间的个数就是整个函数要返回的值。用右边界固定的前提下，左边界最左边的下标去计算也是完全可以的。
            count += (r - l);
        }
        return count;
    }

    private void put(int val) {
        if (rcd[val] == 0) {
            ++ size;
        }
        ++ rcd[val];
    }

    private void remove(int val) {
        -- rcd[val];
        if (rcd[val] == 0) {
            -- size;
        }
    }
}
```

#### [632. 最小区间](https://leetcode-cn.com/problems/smallest-range-covering-elements-from-k-lists/)

题目：

你有 k 个 非递减排列 的整数列表。找到一个 最小 区间，使得 k 个列表中的每个列表至少有一个数包含在其中。

我们定义如果 b-a < d-c 或者在 b-a == d-c 时 a < c，则区间 [a,b] 比 [c,d] 小。



解答：

这道题有两种解法，一种是直接上滑动窗口，记录窗口内的元素所属数组，统计个数，达到了k即可，最后进行比较长度；第二种是优先队列，即合并K个有序链表，通过不断取得某一数组最小值，维持整个优先队列大小为k，即里面的元素均来自不同的数组，然后计算最大元素与最小元素差值，找到最小的即可。最大值通过max维护，最小值通过队首元素维护。

``` Java
public int[] smallestRange(List<List<Integer>> nums) {
    int n = nums.size();
    int inf = 0x3f3f3f;
    int max = -inf; // 当前最大值
    int st = -inf;  // 起点
    int ed = inf;   // 终点

    PriorityQueue<Node> pq = new PriorityQueue<>((o1, o2) -> Integer.compare(o1.val, o2.val));

    // 相当于合并k个有序链表，把 head 放进去
    for (int i = 0; i < n; i++) {
        int val = nums.get(i).get(0);
        pq.offer(new Node(i, 0, val));
        max = Math.max(max, val);
    }
    
    // 必须包含 k 个元素
    while (pq.size() == n) {
        Node node = pq.poll();
        int i = node.i;
        int j = node.j;
        int val = node.val;

        // 更新区间长度
        if (max - val < ed - st) {
            st = val;
            ed = max;
        }
        
        // 为堆中填充元素
        if (j + 1 < nums.get(i).size()) {
            int nVal = nums.get(i).get(j + 1);
            pq.offer(new Node(i, j + 1, nVal));
            max = Math.max(max, nVal);
        }
    }
    return new int[]{st, ed};

}

class Node{
    int i, j, val;

    public Node(int i, int j, int val) {
        this.i = i;
        this.j = j;
        this.val = val;
    }
}
```

#### [1312. 让字符串成为回文串的最少插入次数](https://leetcode-cn.com/problems/minimum-insertion-steps-to-make-a-string-palindrome/)

题目：

给你一个字符串 `s` ，每一次操作你都可以在字符串的任意位置插入任意字符。

请你返回让 `s` 成为回文串的 **最少操作次数** 。



解答：

这道题可以使用贪心思想，即，尽可能利用已经存在的**回文子序列**，这样添加操作就只是围绕这个子序列进行对称处理，所以变成了求最长回文子序列问题，这个问题还是简单的。

``` Java
class Solution {
    public int minInsertions(String s) {
        char[] str = s.toCharArray();
        int[][] dp = new int[str.length][str.length];
        for (int i = 0; i < str.length; ++ i) {
            dp[i][i] = 1;
        }
        if (str.length == 1) {
            return 0;
        }
        int max = 0;
        for (int i = str.length - 2; i >= 0; -- i) {
            for (int j = i + 1; j < str.length; ++ j) {
                if (str[i] == str[j]) {
                    dp[i][j] = dp[i + 1][j - 1] + 2;
                } else {
                    dp[i][j] = Math.max(dp[i + 1][j], dp[i][j - 1]);
                }
                max = Math.max(max, dp[i][j]);
            }
        }
        return str.length - max;
    }
}
```

#### [154. 寻找旋转排序数组中的最小值 II](https://leetcode-cn.com/problems/find-minimum-in-rotated-sorted-array-ii/)

题目：

已知一个长度为 n 的数组，预先按照升序排列，经由 1 到 n 次 旋转 后，得到输入数组。例如，原数组 nums = [0,1,4,4,5,6,7] 在变化后可能得到：
若旋转 4 次，则可以得到 [4,5,6,7,0,1,4]
若旋转 7 次，则可以得到 [0,1,4,4,5,6,7]
注意，数组 [a[0], a[1], a[2], ..., a[n-1]] 旋转一次 的结果为数组 [a[n-1], a[0], a[1], a[2], ..., a[n-2]] 。

给你一个可能存在 重复 元素值的数组 nums ，它原来是一个升序排列的数组，并按上述情形进行了多次旋转。请你找出并返回数组中的 最小元素 。



解答：

最简单粗暴的方法：直接遍历。但是肯定没这么简单，所以这题考的是二分，每次判断二分的位置是否是递增，还是递减，还是出现了转折点，即可。

``` Java
func findMin(nums []int) int {
	index := -1
	l := len(nums)
	for i := 0; i < l-1; i++ {
		if nums[i] > nums[i+1] {
			index = i
			break
		}
	}
	if index == -1 {
		if nums[l-1] < nums[0] {
			return nums[l-1]
		} else {
			return nums[0]
		}
	}
	return nums[index+1]
}
```

#### [403. 青蛙过河](https://leetcode-cn.com/problems/frog-jump/)

题目：

一只青蛙想要过河。 假定河流被等分为若干个单元格，并且在每一个单元格内都有可能放有一块石子（也有可能没有）。 青蛙可以跳上石子，但是不可以跳入水中。

给你石子的位置列表 stones（用单元格序号 升序 表示）， 请判定青蛙能否成功过河（即能否在最后一步跳至最后一块石子上）。

开始时， 青蛙默认已站在第一块石子上，并可以假定它第一步只能跳跃一个单位（即只能从单元格 1 跳至单元格 2 ）。

如果青蛙上一步跳跃了 k 个单位，那么它接下来的跳跃距离只能选择为 k - 1、k 或 k + 1 个单位。 另请注意，青蛙只能向前方（终点的方向）跳跃。



解答：

暴力模拟：记录每次可以到达的位置，然后记录这个位置能跳跃的步数，最后看能不能抵达即可。

``` Java
class Solution {
    private static class Pair {
      	// 位置下标
        int index;
      	// 指出这个位置的步数
        HashSet<Integer> hashSet = new HashSet<>();
    }

    public boolean canCross(int[] stones) {
        HashMap<Integer, Integer> indexMap = new HashMap<>();
        Pair[] pairs = new Pair[stones.length];
        for (int i = 0; i < stones.length; ++i) {
            pairs[i] = new Pair();
            pairs[i].index = stones[i];
            indexMap.put(stones[i], i);
        }
        pairs[0].hashSet.add(1);
        for (int i = 0; i < stones.length - 1; ++i) {
          	// 看一下当前位置能够到达的所有位置
            for (int step : pairs[i].hashSet) {
                if (step == 0) continue;
                int tmp = step + pairs[i].index;
                // System.out.println(i);
              	// 如果存在这样的位置，则更新这个位置能够跳的步数
                if (indexMap.containsKey(tmp)) {
                    int ii = indexMap.get(tmp);
                    pairs[ii].hashSet.add(step - 1);
                    pairs[ii].hashSet.add(step);
                    pairs[ii].hashSet.add(step + 1);
                }
            }
        }
        return !pairs[stones.length - 1].hashSet.isEmpty();
    }
}
```

#### [72. 编辑距离](https://leetcode-cn.com/problems/edit-distance/)

题目：

给你两个单词 word1 和 word2，请你计算出将 word1 转换成 word2 所使用的最少操作数 。

你可以对一个单词进行如下三种操作：

插入一个字符
删除一个字符
替换一个字符



解答：

典型的DP问题，其实这题乍一看我还是蛮蒙的，但是很多时候就硬DP，所以有dp\[i]\[j]为world1前i个与world2前j个转换所需的最少次数。

当他们不同时，