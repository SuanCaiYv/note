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

当他们不同时，无非就是删除：dp\[i]\[j-1] + 1；修改：dp\[i-1]\[j-1]+1；添加：dp\[i-1]\[j]+1之中选择。

``` Java
func min(x, y int) int {
	if x < y {
		return x
	} else {
		return y
	}
}

func minDistance(word1 string, word2 string) int {
	str1 := []byte(word1)
	str2 := []byte(word2)
	len1 := len(str1)
	len2 := len(str2)
	var dp [510][510]int
	if len1 == 0 {
		return len2
	} else if len2 == 0 {
		return len1
	} else {
		for i := 1; i <= len1; i ++ {
			dp[i][0] = i
		}
		for i := 1; i <= len2; i ++ {
			dp[0][i] = i
		}
	}
	for i := 1; i <= len1; i ++ {
		for j := 1; j <= len2; j++ {
			if str1[i-1] == str2[j-1] {
				dp[i][j] = dp[i-1][j-1]
			} else {
                // 在不同时，选择编辑/添加/删除选一个
				dp[i][j] = min(dp[i-1][j-1], min(dp[i-1][j], dp[i][j-1])) + 1
			}
		}
	}
	return dp[len1][len2]
}
```

#### [1269. 停在原地的方案数](https://leetcode-cn.com/problems/number-of-ways-to-stay-in-the-same-place-after-some-steps/)

题目：

有一个长度为 arrLen 的数组，开始有一个指针在索引 0 处。

每一步操作中，你可以将指针向左或向右移动 1 步，或者停在原地（指针不能被移动到数组范围外）。

给你两个整数 steps 和 arrLen ，请你计算并返回：在恰好执行 steps 次操作以后，指针仍然指向索引 0 处的方案数。

由于答案可能会很大，请返回方案数 模 10^9 + 7 后的结果。



解答：DP大法好啊！直接定义dp\[i]\[j]表示移动了i次，在j位置的方案数。则dp\[i]\[j] = dp\[i-1]\[j] + dp\[i-1]\[j-1] + dp\[i-1]\[j+1]；此时注意边界检查。

``` Java
class Solution {
    public int numWays(int steps, int arrLen) {
        final int MODULO = 1000000007;
        int maxColumn = Math.min(arrLen - 1, steps);
        int[][] dp = new int[steps + 1][maxColumn + 1];
        dp[0][0] = 1;
        for (int i = 1; i <= steps; i++) {
            for (int j = 0; j <= maxColumn; j++) {
                dp[i][j] = dp[i - 1][j];
                if (j - 1 >= 0) {
                    dp[i][j] = (dp[i][j] + dp[i - 1][j - 1]) % MODULO;
                }
                if (j + 1 <= maxColumn) {
                    dp[i][j] = (dp[i][j] + dp[i - 1][j + 1]) % MODULO;
                }
            }
        }
        return dp[steps][0];
    }
}
```

#### [42. 接雨水](https://leetcode-cn.com/problems/trapping-rain-water/)

题目：

给定 *n* 个非负整数表示每个宽度为 1 的柱子的高度图，计算按此排列的柱子，下雨之后能接多少雨水。

 

**示例 1：**

![img](https://assets.leetcode-cn.com/aliyun-lc-upload/uploads/2018/10/22/rainwatertrap.png)



解答：

不算很典型的单调栈题目，因为想要接住雨水，必须找到后面一个大于等于它的柱子才行，而找到第一个大于等于/小于等于/大于/小于场景正是单调栈可以做的事，唯一麻烦的就是处理起来细节比较繁琐，因为我们不能单纯地找到柱子，还要计算间距和高度差。

``` Java
class Solution {
    private static class Pair {
        int index;
        int value;
    }

    private final LinkedList<Pair> list = new LinkedList<>();

    private int ans = 0;

    private void put(Pair pair) {
        if (!list.isEmpty()) {
            Pair first = list.getFirst();
            while (pair.value > first.value) {
                Pair tmp = list.getFirst();
                list.removeFirst();
                first = list.peekFirst();
                if (first == null) {
                    break;
                } else {
                    ans += (Math.min(pair.value, first.value) - tmp.value) * (pair.index - first.index - 1);
                }
            }
        }
        list.addFirst(pair);
    }

    public int trap(int[] height) {
        Pair[] pairs = new Pair[height.length];
        int begin = -1;
        for (int i = 0; i < height.length; ++i) {
            pairs[i] = new Pair();
            pairs[i].index = i;
            pairs[i].value = height[i];
            if (height[i] != 0 && begin == -1) {
                begin = i;
            }
        }
        if (begin == -1) {
            return 0;
        }
        for (int i = begin; i < height.length; ++i) {
            put(pairs[i]);
        }
        return ans;
    }
}
```

#### [25. K 个一组翻转链表](https://leetcode-cn.com/problems/reverse-nodes-in-k-group/)

题目：

给你一个链表，每 k 个节点一组进行翻转，请你返回翻转后的链表。

k 是一个正整数，它的值小于或等于链表的长度。

如果节点总数不是 k 的整数倍，那么请将最后剩余的节点保持原有顺序。

进阶：

你可以设计一个只使用常数额外空间的算法来解决此问题吗？
你不能只是单纯的改变节点内部的值，而是需要实际进行节点交换。

**示例 1：**

![img](https://assets.leetcode.com/uploads/2020/10/03/reverse_ex1.jpg)



解答：

模拟题，就硬模拟。

``` Java
class Solution {

    public ListNode reverseKGroup(ListNode head, int k) {
        ListNode root = new ListNode();
        ListNode tmp = root;
        int i = k;
        while (head != null) {
            ListNode from = head, to = null;
            while (head != null && i > 0) {
                to = head;
                head = head.next;
                -- i;
            }
            if (head == null && i > 0) {
                root.next = from;
            } else {
                to.next = null;
                ff(from);
                root.next = to;
                root = from;
            }
            i = k;
        }
        return tmp.next;
    }

    private ListNode ff(ListNode node) {
        if (node.next == null) {
            return node;
        }
        ListNode prev = ff(node.next);
        node.next = null;
        prev.next = node;
        return node;
    }
}
```

#### [剑指 Offer 51. 数组中的逆序对](https://leetcode-cn.com/problems/shu-zu-zhong-de-ni-xu-dui-lcof/)

题目：

在数组中的两个数字，如果前面一个数字大于后面的数字，则这两个数字组成一个逆序对。输入一个数组，求出这个数组中的逆序对的总数。



解答：

这题考的是活用归并排序，基于这样一个思想：划分出mid位置，mid之前全部是有序的，mid后面也是有序的；然后判断，nums\[a]与nums\[b]之间的关系，其中a <= mid < b。如果nums\[a] > nums\[b]，则mid - a这个范围内的数字都可以和nums\[b]凑成逆序对。不过记得在处理过程中同时合并两个有序数组。

``` Java
class Solution {
    public int reversePairs(int[] nums) {
        mergeSort(nums, 0, nums.length - 1);
        return ans;
    }

    private int ans = 0;

    private void mergeSort(int[] nums, int from, int to) {
        if (from >= to) {
            return;
        }
        int mid = (from + to) / 2;
        mergeSort(nums, from, mid);
        mergeSort(nums, mid + 1, to);
        int a = from, b = mid + 1;
        int k = 0;
        int[] tmp = new int[to - from + 1];
        while (a <= mid) {
            while (b <= to) {
                if (nums[b] < nums[a]) {
                    // System.out.println(mid + " " + a + " " + b);
                    // System.out.println(nums[a] + " | " + nums[b]);
                    tmp[k++] = nums[b++];
                    ans += (mid - a + 1);
                } else {
                    break;
                }
            }
            tmp[k++] = nums[a++];
        }
        while (b <= to) {
            // System.out.println(from + " # " + to);
            // System.out.println(k + " : " + tmp.length);
            tmp[k++] = nums[b++];
        }
        if (to + 1 - from >= 0) System.arraycopy(tmp, 0, nums, from, to + 1 - from);
    }
}
```

#### [224. 基本计算器](https://leetcode-cn.com/problems/basic-calculator/)

题目：

给你一个字符串表达式 `s` ，请你实现一个基本计算器来计算并返回它的值。



解答：

模拟题，和计算逆波兰表达式有点类似，通过栈解决。

``` Java
class Solution {
    private final LinkedList<Integer> numbers = new LinkedList<>();

    private final LinkedList<Character> operators = new LinkedList<>();

    private Item[] items;

    private static class Item {
        private int num;

        private char op;

        private boolean type;
    }

    public int calculate(String s) {
        char[] tmpStr = s.toCharArray();
        int len = 0;
        for (int i = 0; i < tmpStr.length; ++ i) {
            if (tmpStr[i] != ' ') {
                tmpStr[len++] = tmpStr[i];
            }
        }
        String input = new String(tmpStr, 0, len);
        if (input.charAt(0) == '-') {
            input = "0" + input;
        }
        String s1 = "\\(-", s2 = "(0-";
        String s3 = "\\(+", s4 = "(0";
        input = input.replaceAll(s1, s2).replaceAll(s3, s4);
        char[] str = input.toCharArray();
        convert(str, input);
        for (int i = 0; i < items.length;) {
            if (!items[i].type) {
                putNum(items[i++].num);
            } else {
                putOps(items[i++].op);
                if (i < items.length && !items[i].type) {
                    putNum(items[i++].num);
                }
            }
            // log();
        }
        compute();
        return numbers.getFirst();
    }

    @SuppressWarnings("unchecked")
    private void log() {
        int len = Math.max(numbers.size(), operators.size());
        var a = (LinkedList<Integer>) numbers.clone();
        var b = (LinkedList<Character>) operators.clone();
        for (int i = 0; i < len; ++ i) {
            if (i >= numbers.size()) {
                System.out.println("      " + b.getFirst());
                b.removeFirst();
            } else if (i >= operators.size()) {
                System.out.println(a.getFirst() + "      ");
                a.removeFirst();
            } else {
                System.out.println(a.getFirst() + "     " + b.getFirst());
                a.removeFirst();
                b.removeFirst();
            }
        }
        System.out.println("#######");
    }

    private void convert(char[] str, String s) {
        Item[] tmpItems = new Item[str.length];
        int len = 0;
        for (int i = 0; i < str.length;) {
            Item item = new Item();
            if (str[i] >= '0' && str[i] <= '9') {
                int k = i;
                while (i < str.length && str[i] >= '0' && str[i] <= '9') ++ i;
                item.num = Integer.parseInt(s.substring(k, i));
                item.type = false;
                tmpItems[len++] = item;
            } else {
                item.op = str[i++];
                item.type = true;
                tmpItems[len++] = item;
            }
        }
        items = new Item[len];
        System.arraycopy(tmpItems, 0, items, 0, len);
    }

    private void compute() {
        char top;
        int v2, v1;
        boolean flag = false;
        while (!operators.isEmpty()) {
            top = operators.getFirst();
            if (top == '(') {
                if (flag) {
                    operators.removeFirst();
                }
            } else {
                operators.removeFirst();
            }
            if (top == ')') {
                flag = true;
                continue;
            }
            if (top == '(') {
                break;
            }
            v2 = numbers.getFirst();
            numbers.removeFirst();
            v1 = numbers.getFirst();
            numbers.removeFirst();
            // System.out.println("c:" + v1 + " " + top + " " + v2);
            int v;
            switch (top) {
                case '+' -> v = v1 + v2;
                case '-' -> v = v1 - v2;
                case '*' -> v = v1 * v2;
                case '/' -> v = v1 / v2;
                default -> v = 0;
            }
            numbers.addFirst(v);
        }
    }

    private void putNum(int num) {
        numbers.addFirst(num);
    }

    private void putOps(char op) {
        if (operators.isEmpty()) {
            operators.addFirst(op);
            // log();
        } else {
            if (op == '(') {
                operators.addFirst(op);
                // log();
            } else if (op == ')') {
                operators.addFirst(op);
                // log();
                compute();
            } else {
                if (lower(op, operators.getFirst())) {
                    operators.addFirst(op);
                    // log();
                } else {
                    // log();
                    compute();
                    operators.addFirst(op);
                }
            }
        }
    }

    private boolean lower(char a, char b) {
        if (a == '*' || a == '/') {
            return b == '+' || b == '-';
        } else {
            return false;
        }
    }
}
```

#### [124. 二叉树中的最大路径和](https://leetcode-cn.com/problems/binary-tree-maximum-path-sum/)

题目：

路径 被定义为一条从树中任意节点出发，沿父节点-子节点连接，达到任意节点的序列。同一个节点在一条路径序列中 至多出现一次 。该路径 至少包含一个 节点，且不一定经过根节点。

路径和 是路径中各节点值的总和。

给你一个二叉树的根节点 root ，返回其 最大路径和 。



解答：

就硬遍历，硬模拟。

``` Java
class Solution {
    private int ans = Integer.MIN_VALUE;

    public int maxPathSum(TreeNode root) {
        find(root);
        return ans;
    }

    private int find(TreeNode parent) {
        if (parent == null) {
            return Integer.MIN_VALUE;
        }
        ans = Math.max(ans, parent.val);
        int left = find(parent.left);
        int right = find(parent.right);
        if (parent.val >= 0) {
            if (left >= 0) {
                if (right >= 0) {
                    ans = Math.max(ans, parent.val + left + right);
                    return parent.val + Math.max(left, right);
                } else {
                    ans = Math.max(ans, parent.val + left);
                    return parent.val + left;
                }
            } else {
                if (right >= 0) {
                    ans = Math.max(ans, parent.val + right);
                    return parent.val + right;
                } else {
                    ans = Math.max(ans, parent.val);
                    return parent.val;
                }
            }
        } else {
            if (left >= 0) {
                if (right >= 0) {
                    ans = Math.max(ans, left);
                    ans = Math.max(ans, right);
                    ans = Math.max(ans, parent.val + left + right);
                    int v1 = parent.val + left;
                    int v2 = parent.val + right;
                    int max = Math.max(v1, v2);
                    return max;
                } else {
                    ans = Math.max(ans, left);
                    int v = parent.val + left;
                    return v;
                }
            } else {
                if (right >= 0) {
                    ans = Math.max(ans, right);
                    int v = parent.val + right;
                    return v;
                } else {
                    return parent.val;
                }
            }
        }
    }
}
```

#### [32. 最长有效括号](https://leetcode-cn.com/problems/longest-valid-parentheses/)

题目：

给你一个只包含 `'('` 和 `')'` 的字符串，找出最长有效（格式正确且连续）括号子串的长度。



解答：

这道题还是很有意思的，其实一开始大家都想得到使用栈，这没错，但是我看评论区才发现我们可以通过栈，把所有合法的括号`(` `)`标记成1，然后统计连续的1的长度即可。

``` Java
class Solution {
    private static class Pair {
        char val;
        int index;
    }

    public int longestValidParentheses(String s) {
        char[] str = s.toCharArray();
        int[] dp = new int[str.length];
        Pair[] pairs = new Pair[str.length];
        LinkedList<Pair> stack = new LinkedList<>();
        for (int i = 0; i < str.length; ++ i) {
            pairs[i] = new Pair();
            pairs[i].val = str[i];
            pairs[i].index = i;
        }
        for (Pair pair : pairs) {
            if (pair.val == '(') {
                stack.addFirst(pair);
            } else {
                if (!stack.isEmpty()) {
                    Pair tmp = stack.getFirst();
                    stack.removeFirst();
                    dp[tmp.index] = 1;
                    dp[pair.index] = 1;
                }
            }
        }
        int ans = 0, count = 0;
        for (int a : dp) {
            if (a == 1) {
                ++ count;
            } else {
                count = 0;
            }
            ans = Math.max(ans, count);
        }
        return ans;
    }
}
```

#### [664. 奇怪的打印机](https://leetcode-cn.com/problems/strange-printer/)

题目：

有台奇怪的打印机有以下两个特殊要求：

* 打印机每次只能打印由 同一个字符 组成的序列。
* 每次可以在任意起始和结束位置打印新字符，并且会覆盖掉原来已有的字符。

给你一个字符串 s ，你的任务是计算这个打印机打印它需要的最少打印次数。



解答：

这道题其实如果可以想到DP还是很简单的，直接二维DP，i, j表示从i到j位置的字符串的打印次数。其实到这里我们可以总结一下，一般涉及到字符串DP的题目，都是内循环，且如果涉及到最值问题，一般都是DP，区间最值则是拆分成循环比较。



内循环：

``` Java
for (int i = 0; i < len; ++ i) {
    for (int j = i; j >= 0; -- j) {
      	dp[j][i] = ...
    }
}
```

切分求值：

``` Java
for (int i = 0; i < len; ++ i) {
    for (int j = i; j >= 0; -- j) {
      	for (int k = i-1; k >= j; -- k) {
          dp[j][i] = max/min(dp[j][k], dp[k+1][i]);
        }
    }
}
```



所以我们有题解：

``` Java
class Solution {

    public int strangePrinter(String s) {
        char[] str = s.toCharArray();
        int[][] dp = new int[str.length + 1][str.length + 1];
        for (int i = 0; i < str.length; ++ i) {
            dp[i][i] = 1;
        }
        dp[0][0] = 1;
        for (int i = 0; i < str.length; ++ i) {
            for (int j = i-1; j >= 0; -- j) {
                if (str[i] == str[j]) {
                    dp[j][i] = dp[j][i-1];
                } else {
                    dp[j][i] = Integer.MAX_VALUE;
                    for (int k = i-1; k >= j; -- k) {
                        // 字符串一般使用切分DP求最值
                        dp[j][i] = Math.min(dp[j][i], dp[j][k] + dp[k+1][i]);
                    }
                }
            }
        }
        return dp[0][str.length-1];
    }
}
```

#### [140. 单词拆分 II](https://leetcode-cn.com/problems/word-break-ii/)

题目：

给定一个非空字符串 s 和一个包含非空单词列表的字典 wordDict，在字符串中增加空格来构建一个句子，使得句子中所有的单词都在词典中。返回所有这些可能的句子。

说明：

分隔时可以重复使用字典中的单词。
你可以假设字典中没有重复的单词。



解答：

这道题其实很简单，就是模拟题，但是为了优化我们提前记录一下，哪些区间的单词是存在字典中的，然后回溯处理即可。

``` Java
class Solution {

    private final HashSet<String> hashSet = new HashSet<>();

    private String str;

    private boolean[][] dp;

    public List<String> wordBreak(String s, List<String> wordDict) {
        str = s;
        dp = new boolean[s.length()][s.length()];
        hashSet.addAll(wordDict);
        for (int i = 0; i < s.length(); ++ i) {
            for (int j = i; j < s.length(); ++ j) {
                String tmpStr = s.substring(i, j + 1);
                if (hashSet.contains(tmpStr)) {
                    // System.out.println(i + " : " + j);
                    dp[i][j] = true;
                } else {
                    dp[i][j] = false;
                }
            }
        }
        List<List<String>> ans = f(0, s.length() - 1);
        if (ans == null) {
            return new LinkedList<>();
        }
        List<String> ans0 = new LinkedList<>();
        for (List<String> a : ans) {
            StringBuilder stringBuilder = new StringBuilder();
            for (String b : a) {
                stringBuilder.append(b).append(" ");
            }
            stringBuilder.deleteCharAt(stringBuilder.length() - 1);
            ans0.add(stringBuilder.toString());
        }
        return ans0;
    }

    public List<List<String>> f(int from, int to) {
        if (from > to) {
            return null;
        }
        List<List<String>> ans = new LinkedList<>();
        for (int i = from; i <= to; ++ i) {
            if (i == to) {
                if (dp[from][to]) {
                    // 特殊处理一下，因为此时就不需要递归了。
                    List<String> tmp = new LinkedList<>();
                    tmp.add(str.substring(from, to + 1));
                    ans.add(tmp);
                }
            } else {
                if (dp[from][i]) {
                    List<List<String>> tmpList = f(i + 1, to);
                    if (tmpList != null) {
                        for (List<String> a : tmpList) {
                            List<String> tmp = new LinkedList<>();
                            tmp.add(str.substring(from, i + 1));
                            tmp.addAll(a);
                            ans.add(tmp);
                        }
                    }
                }
            }
        }
        if (ans.isEmpty()) {
            return null;
        } else {
            return ans;
        }
    }
}
```

#### 