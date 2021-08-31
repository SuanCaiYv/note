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

这道题有两种解法，一种是直接上滑动窗口，记录窗口内的元素所属数组，统计个数，达到了k即可，最后进行比较长度；第二种是优先队列，即**合并K个有序链表**，通过不断取得某一数组最小值，维持整个优先队列大小为k，即里面的元素均来自不同的数组，然后计算最大元素与最小元素差值，找到最小的即可。最大值通过max维护，最小值通过队首元素维护。

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

#### [879. 盈利计划](https://leetcode-cn.com/problems/profitable-schemes/)

题目：

集团里有 n 名员工，他们可以完成各种各样的工作创造利润。

第 i 种工作会产生 profit[i] 的利润，它要求 group[i] 名成员共同参与。如果成员参与了其中一项工作，就不能参与另一项工作。

工作的任何至少产生 minProfit 利润的子集称为 盈利计划 。并且工作的成员总数最多为 n 。

有多少种计划可以选择？因为答案很大，所以 返回结果模 10^9 + 7 的值。



解答：

最值问题，和选取问题，一般都是背包问题，这题是很典型的三维背包问题，我在这里放一个背包通项：

``` Java
public class Bag {

    // 01背包
    private int package01(int w, int[] ws, int[] vs) {
      	// 从前i个中选，当重量不超过j时的最大价值
        int[][] dp1 = new int[ws.length][w + 1];
        for (int i = w; i >= ws[0]; -- i) {
            dp1[0][i] = vs[0];
        }
        for (int i = 1; i < ws.length; ++ i) {
            // 个人感觉，这里正序倒序应该都可以
            for (int j = w; j >= 0; --j) {
                // if ...
                dp1[i][j] = Math.max(dp1[i-1][j], dp1[i-1][j-ws[i]] + vs[i]);
            }
        }
        // 化为一维
        int[] dp2 = new int[w + 1];
        for (int i = 0; i < ws.length; ++ i) {
            for (int j = w; j >= 0; -- j) {
                // if ...
                dp2[j] = Math.max(dp2[j], dp2[j-ws[i]] + vs[i]);
            }
        }
        return dp2[w];
    }

    // 完全背包
    private int packageMulti(int w, int[] ws, int[] vs) {
        int[][] dp1 = new int[ws.length][w + 1];
        for (int j = ws[0]; j <= w; ++ j) {
            dp1[0][j] = Math.max(dp1[0][j], dp1[0][j-ws[0]] + vs[0]);
        }
        for (int i = 1; i < ws.length; ++ i) {
            // 为什么这里是正序，其实很好理解，就是我的dp[i][j]依赖于dp[i][k]的结果，这里的k < j，所以我们只能是正序
            for (int j = 0; j <= w; ++ j) {
                dp1[i][j] = Math.max(dp1[i-1][j], dp1[i][j-ws[i]] + vs[i]);
            }
        }
        int[] dp2 = new int[w + 1];
        for (int i = 0; i < ws.length; ++ i) {
            for (int j = 0; j <= w; ++ j) {
                dp2[j] = Math.max(dp2[j], dp2[j-ws[i]] + vs[i]);
            }
        }
        return dp2[w];
    }
}
```

一般来说，01背包强调要么不选，要么只选一个；完全背包强调要么不选，要么可以选无数个。此外如果每个物品个数都有限制，那只能再来一个循环进行判断，没有什么好的解法。



此外，背包问题的循环处理，一般会把限制层放在内层，也就是把容量，重量，成本等限制选择的循环放在第二个循环。



现在我们来看看这题解答：

``` Java
class Solution {
    public int profitableSchemes(int n, int minProfit, int[] group, int[] profit) {
        int mod = (int) 1e9+7;
        // 利润不大于minProfit的方案数
        int[][][] dp = new int[110][110][110];
        for (int i = 0; i <= group.length; ++ i) {
            for (int j = 0; j <= n; ++ j) {
                dp[i][j][0] = 1;
            }
        }
        for (int i = 1; i <= group.length; ++ i) {
            for (int j = 0; j <= n; ++ j) {
                for (int k = 0; k <= minProfit; ++ k) {
                    if (j >= group[i-1]) {
                        if (k >= profit[i-1]) {
                            dp[i][j][k] = (dp[i-1][j][k] + dp[i-1][j-group[i-1]][k-profit[i-1]]) % mod;
                        } else {
                            dp[i][j][k] = (dp[i-1][j][k] + dp[i-1][j-group[i-1]][0]) % mod;
                        }
                    } else {
                        dp[i][j][k] = dp[i-1][j][k];
                    }
                }
            }
        }
        return dp[group.length][n][minProfit];
    }
}
```

#### [1449. 数位成本和为目标值的最大数字](https://leetcode-cn.com/problems/form-largest-integer-with-digits-that-add-up-to-target/)

题目：

给你一个整数数组 cost 和一个整数 target 。请你返回满足如下规则可以得到的 最大 整数：

给当前结果添加一个数位（i + 1）的成本为 cost[i] （cost 数组下标从 0 开始）。
总成本必须恰好等于 target 。
添加的数位中没有数字 0 。
由于答案可能会很大，请你以字符串形式返回。

如果按照上述要求无法得到任何整数，请你返回 "0" 。



解答：

这道题很容易看出来是完全背包问题。现在我们来看看该题：

``` Java
class Solution {
    private int compare(String a, String b) {
        if (a.charAt(0) == '-') {
            return -1;
        } else if (b.charAt(0) == '-') {
            return 1;
        }
        if (a.length() > b.length()) {
            return 1;
        } else if (a.length() < b.length()) {
            return -1;
        } else {
            return a.compareTo(b);
        }
    }

    public String largestNumber(int[] cost, int target) {
        String[] dp = new String[target+1];
        Arrays.fill(dp, "-1");
        dp[0] = "";
        for (int i = 1; i <= cost.length; ++ i) {
            for (int j = 0; j <= target; ++ j) {
                if (j >= cost[i-1]) {
                    String prev = dp[j];
                    String tmpStr = dp[j-cost[i-1]];
                    String tmpS = i + "";
                    if (tmpStr.length() == 0) {
                        dp[j] = tmpS;
                    } else {
                        String s1 = tmpS + tmpStr;
                        String s2 = tmpStr + tmpS;
                        if (s1.charAt(0) != '-' && s2.charAt(0) != '-') {
                            if (compare(s1, s2) > 0) {
                                if (compare(s1, prev) > 0) {
                                    dp[j] = s1;
                                }
                            } else {
                                if (compare(s2, prev) > 0) {
                                    dp[j] = s2;
                                }
                            }
                        }
                    }
                }
                // System.out.print(dp[j] + " ");
            }
            // System.out.println();
        }
        return dp[target].charAt(0) == '-' ? "0" : dp[target];
    }
}
```

#### [127. 单词接龙](https://leetcode-cn.com/problems/word-ladder/)

题目：

字典 wordList 中从单词 beginWord 和 endWord 的 转换序列 是一个按下述规格形成的序列：

序列中第一个单词是 beginWord 。
序列中最后一个单词是 endWord 。
每次转换只能改变一个字母。
转换过程中的中间单词必须是字典 wordList 中的单词。
给你两个单词 beginWord 和 endWord 和一个字典 wordList ，找到从 beginWord 到 endWord 的 最短转换序列 中的 单词数目 。如果不存在这样的转换序列，返回 0。



解答：

这道题是很典型的BFS搜索问题，因为每次只能改变一个字母，而一共有26个字母，我们就可以对单词的每个字母进行从a-z的替换，然后寻找最短路径。



其实说到最短路径，第一时间肯定还是想到了Dijkstra算法，但是当路径权值相同时，Dijkstra算法就可以退化成BFS，而BFS肯定写起来快，所以我们这题就是这么干的。此外，既然提到了Dijkstra，我们就来理一下最短路算法：



* 设置起始位置为第一个节点。
* 遍历所有未标记节点，找到离起始节点最近的节点k。
* 以k为中心，通过k更新它所能到达的节点与起始节点的距离
* 把k标记为以处理。
* 重复第二步。



然后此时我们来看看这题，使用BFS处理：

``` Java
class Solution {
    public int ladderLength(String beginWord, String endWord, List<String> wordList) {
        HashMap<String, Integer> dp = new HashMap<>(wordList.size() + 2);
        HashMap<String, Boolean> rcd = new HashMap<>(wordList.size() + 2);
        for (String s : wordList) {
            dp.put(s, Integer.MAX_VALUE);
            rcd.put(s, false);
        }
        if (!dp.containsKey(endWord)) {
            return 0;
        }
        dp.put(endWord, Integer.MAX_VALUE);
        rcd.put(endWord, false);
        dp.put(beginWord, 1);
        rcd.put(beginWord, false);
        LinkedList<String> queue = new LinkedList<>();
        queue.addFirst(beginWord);
        while (!queue.isEmpty()) {
            String first = queue.getFirst();
            rcd.put(first, true);
            queue.removeFirst();
            char[] str = first.toCharArray();
            for (int i = 0; i < str.length; ++ i) {
                char origin = str[i];
                for (int j = 0; j < 26; ++ j) {
                    if ('a' + j == origin) {
                        continue;
                    }
                    str[i] = (char) ('a' + j);
                    String tmpStr = new String(str);
                    if (dp.containsKey(tmpStr)) {
                        dp.put(tmpStr, Math.min(dp.get(first) + 1, dp.get(tmpStr)));
                        if (!rcd.get(tmpStr)) {
                            rcd.put(tmpStr, true);
                            queue.addLast(tmpStr);
                        }
                        // System.out.println(tmpStr + "," + dp.get(tmpStr));
                    }
                }
                str[i] = origin;
            }
        }
        return dp.get(endWord) == Integer.MAX_VALUE ? 0 : dp.get(endWord);
    }
}
```

#### [126. 单词接龙 II](https://leetcode-cn.com/problems/word-ladder-ii/)

题目：

按字典 wordList 完成从单词 beginWord 到单词 endWord 转化，一个表示此过程的 转换序列 是形式上像 beginWord -> s1 -> s2 -> ... -> sk 这样的单词序列，并满足：

每对相邻的单词之间仅有单个字母不同。
转换过程中的每个单词 si（1 <= i <= k）必须是字典 wordList 中的单词。注意，beginWord 不必是字典 wordList 中的单词。
sk == endWord
给你两个单词 beginWord 和 endWord ，以及一个字典 wordList 。请你找出并返回所有从 beginWord 到 endWord 的 最短转换序列 ，如果不存在这样的转换序列，返回一个空列表。每个序列都应该以单词列表 [beginWord, s1, s2, ..., sk] 的形式返回。



解答：

这题和上一题最大的不同是，要求记录路径。一般对于最点路径走法的记录，我们使用倒叙记录，即记录当前节点是从哪个节点过来的；然后复原的话，反过来找到路径即可。

``` Java
class Solution {
    List<List<String>> ans = new LinkedList<>();

    String bgnWord;

    HashMap<String, LinkedList<String>> paths;

    public List<List<String>> findLadders(String beginWord, String endWord, List<String> wordList) {
        HashMap<String, Integer> dp = new HashMap<>(wordList.size() + 2);
        HashMap<String, Boolean> rcd = new HashMap<>(wordList.size() + 2);
        paths = new HashMap<>(wordList.size() + 2);
        bgnWord = beginWord;
        for (String s : wordList) {
            dp.put(s, Integer.MAX_VALUE);
            rcd.put(s, false);
            paths.put(s, new LinkedList<>());
        }
        if (!dp.containsKey(endWord)) {
            return new LinkedList<>();
        }
        dp.put(endWord, Integer.MAX_VALUE);
        rcd.put(endWord, false);
        paths.put(endWord, new LinkedList<>());
        dp.put(beginWord, 1);
        rcd.put(beginWord, false);
        paths.put(beginWord, new LinkedList<>());
        LinkedList<String> queue = new LinkedList<>();
        queue.addFirst(beginWord);
        while (!queue.isEmpty()) {
            String first = queue.getFirst();
            rcd.put(first, true);
            queue.removeFirst();
            char[] str = first.toCharArray();
            for (int i = 0; i < str.length; ++ i) {
                char origin = str[i];
                for (int j = 0; j < 26; ++ j) {
                    if ('a' + j == origin) {
                        continue;
                    }
                    str[i] = (char) ('a' + j);
                    String tmpStr = new String(str);
                    if (dp.containsKey(tmpStr)) {
                        int a = dp.get(first) + 1;
                        int b = dp.get(tmpStr);
                        dp.put(tmpStr, Math.min(a, b));
                        if (!rcd.get(tmpStr)) {
                            rcd.put(tmpStr, true);
                            queue.addLast(tmpStr);
                        }
                        if (a < b) {
                            paths.get(tmpStr).clear();
                            paths.get(tmpStr).addLast(first);
                        } else if (a == b) {
                            paths.get(tmpStr).addLast(first);
                        }
                    }
                }
                str[i] = origin;
            }
        }
        LinkedList<String> tmp = new LinkedList<>();
        f(tmp, endWord);
        return ans;
    }

    private void f(LinkedList<String> list, String str) {
        // System.out.println(str + "," + paths.get(str));
        if (str.equals(bgnWord)) {
            list.addFirst(str);
            ans.add((List<String>) list.clone());
            list.removeFirst();
        }
        list.addFirst(str);
        for (String s : paths.get(str)) {
            f(list, s);
        }
        list.removeFirst();
    }
}
```

#### [871. 最低加油次数](https://leetcode-cn.com/problems/minimum-number-of-refueling-stops/)

题目：

汽车从起点出发驶向目的地，该目的地位于出发位置东面 target 英里处。

沿途有加油站，每个 station[i] 代表一个加油站，它位于出发位置东面 station[i][0] 英里处，并且有 station[i][1] 升汽油。

假设汽车油箱的容量是无限的，其中最初有 startFuel 升燃料。它每行驶 1 英里就会用掉 1 升汽油。

当汽车到达加油站时，它可能停下来加油，将所有汽油从加油站转移到汽车中。

为了到达目的地，汽车所必要的最低加油次数是多少？如果无法到达目的地，则返回 -1 。

注意：如果汽车到达加油站时剩余燃料为 0，它仍然可以在那里加油。如果汽车到达目的地时剩余燃料为 0，仍然认为它已经到达目的地。



解答：

这道题其实蛮艹的，因为它解法唯一，只能用优先队列求解，而且思路也是唯一的，作为一个引入优先队列的题目是不错的，但是作为一个通用算法题，可能不是很有意思。



我们通过优先队列，记录汽车行驶过的所有的加油站，如果汽油不够了，我们从经过的所有加油站取一个能加最多油的即可；然后继续。

``` Java
class Solution {
    public int minRefuelStops(int target, int startFuel, int[][] stations) {
        PriorityQueue<Integer> priorityQueue = new PriorityQueue<>((o1, o2) -> -1*Integer.compare(o1, o2));
        int[][] rcd = new int[stations.length + 2][2];
        rcd[0][0] = 0;
        rcd[0][1] = startFuel;
        rcd[rcd.length - 1][0] = target;
        rcd[rcd.length - 1][1] = 0;
        for (int i = 1; i < rcd.length - 1; ++i) {
            rcd[i][0] = stations[i - 1][0];
            rcd[i][1] = stations[i - 1][1];
        }
        int prev = rcd[0][0];
        int curr = startFuel;
        int count = 0;
        for (int i = 1; i < rcd.length; ++ i) {
            int dis = rcd[i][0] - prev;
            curr -= dis;
            // System.out.println(i);
            while (curr < 0) {
                if (priorityQueue.isEmpty()) {
                    return -1;
                }
                curr += priorityQueue.poll();
                // System.out.println("curr: " + curr);
                ++ count;
            }
            priorityQueue.add(rcd[i][1]);
            prev = rcd[i][0];
        }
        return count;
    }
}
```

#### [23. 合并K个升序链表](https://leetcode-cn.com/problems/merge-k-sorted-lists/)

题目：

给你一个链表数组，每个链表都已经按升序排列。

请你将所有链表合并到一个升序链表中，返回合并后的链表。



解答：

优先队列解决，因为涉及到多路排序问题，此时优先队列是一个不错的方案。

``` Java
class Solution {
    public ListNode mergeKLists(ListNode[] lists) {
        PriorityQueue<Integer> priorityQueue = new PriorityQueue<>();
        for (ListNode head : lists) {
            ListNode tmp = head;
            while (tmp != null) {
                priorityQueue.add(tmp.val);
                tmp = tmp.next;
            }
        }
        ListNode listNode = new ListNode();
        ListNode head = listNode;
        ListNode prev = null;
        while (!priorityQueue.isEmpty()) {
            listNode.val = priorityQueue.poll();
            listNode.next = new ListNode();
            prev = listNode;
            listNode = listNode.next;
        }
        if (prev == null) {
            return null;
        } else {
            prev.next = null;
            return head;
        }
    }
}
```

#### [239. 滑动窗口最大值](https://leetcode-cn.com/problems/sliding-window-maximum/)

题目：

给你一个整数数组 nums，有一个大小为 k 的滑动窗口从数组的最左侧移动到数组的最右侧。你只可以看到在滑动窗口内的 k 个数字。滑动窗口每次只向右移动一位。

返回滑动窗口中的最大值。



解答：

这道题涉及到了范围更新，或者动态更新找最值，其实结合[数据流的中位数](https://leetcode-cn.com/problems/find-median-from-data-stream/)这一题，我们基本可以知道，涉及到数据动态更新，求最值的，基本都是滑动窗口。



此题利用了延迟删除的特性，即，获取最大值之前先判断最大值是不是已经不在滑动窗口中了，如果不在我们需要剔除，并获得下一个最大值。关于**延迟删除**，其实还有另一个更加经典的题[滑动窗口的中位数](https://leetcode-cn.com/problems/sliding-window-median/)这一题。

``` Java
class Solution {

    private static class Pair {
        int val;
        int index;

        public Pair() {
        }

        public Pair(int val, int index) {
            this.val = val;
            this.index = index;
        }
    }

    public int[] maxSlidingWindow(int[] nums, int k) {
        PriorityQueue<Pair> priorityQueue = new PriorityQueue<>((v1, v2) -> {
            if (v1.val > v2.val) {
                return -1;
            } else if (v1.val < v2.val) {
                return 1;
            } else {
                // 数值一样时，选择下标最大的，这样可以尽可能用到可用的最大值
                return v1.index > v2.index ? -1 : 0;
            }
        });
        int[] tmp = new int[nums.length];
        int index = 0;
        for (int i = 0; i < k; ++ i) {
            priorityQueue.add(new Pair(nums[i], i));
        }
        tmp[index++] = priorityQueue.peek().val;
        for (int i = k; i < nums.length; ++ i) {
            priorityQueue.add(new Pair(nums[i], i));
            // 去除所有边界之外的最大值
            while (priorityQueue.peek().index <= i-k) {
                priorityQueue.poll();
            }
            tmp[index++] = priorityQueue.peek().val;
        }
        int[] ans = new int[index];
        System.arraycopy(tmp, 0, ans, 0, index);
        return ans;
    }
}
```

#### [407. 接雨水 II](https://leetcode-cn.com/problems/trapping-rain-water-ii/)

题目：

给你一个 `m x n` 的矩阵，其中的值均为非负整数，代表二维高度图每个单元的高度，请计算图中形状最多能接多少体积的雨水。

 

**示例 1:**

![img](https://assets.leetcode.com/uploads/2021/04/08/trap1-3d.jpg)



解答：

二维接雨水，我们用的是单调栈，三维我们想要接住雨水，恐怕需要上优先队列了，啥意思呢？就是说我们此时，想要接住雨水，需要找到最矮的柱子，而这个最矮的柱子就需要优先队列进行查找了。

``` Java
class Solution {

    private int m, n;

    private int[][] heightMap;

    private boolean[][] marked;

    private final int[] X = {1, 0, -1, 0};

    private final int[] Y = {0, 1, 0, -1};

    private int ans = 0;

    public int trapRainWater(int[][] heightMap0) {
        heightMap = heightMap0;
        m = heightMap0.length;
        n = heightMap0[0].length;
        marked = new boolean[m][n];
        if (m < 3 || n < 3) {
            return 0;
        }
        PriorityQueue<Pair> priorityQueue = new PriorityQueue<>(Comparator.comparingInt(p -> p.val));
        int k = Math.max(m, n);
        for (int a = 0; a < k; ++a) {
            // 边缘入
            for (int i = a; i < m - a; ++i) {
                if (i < m && a < n && !marked[i][a]) {
                    priorityQueue.add(new Pair(heightMap[i][a], i, a));
                    marked[i][a] = true;
                }
                if (i < m && n-a-1 < n && n-a-1 >= 0 && !marked[i][n-a-1]) {
                    priorityQueue.add(new Pair(heightMap[i][n - a - 1], i, n - a - 1));
                    marked[i][n - a - 1] = true;
                }
            }
            for (int i = a; i < n - a; ++i) {
                if (a < m && i < n && !marked[a][i]) {
                    priorityQueue.add(new Pair(heightMap[a][i], a, i));
                    marked[a][i] = true;
                }
                if (m-a-1 < m && m-a-1 >= 0 && i < n && !marked[m-a-1][i]) {
                    priorityQueue.add(new Pair(heightMap[m - a - 1][i], m - a - 1, i));
                    marked[m - a - 1][i] = true;
                }
            }
            while (!priorityQueue.isEmpty()) {
                Pair p = priorityQueue.poll();
                // System.out.println(p);
                for (int i = 0; i < 4; ++i) {
                    int x = p.x + X[i];
                    int y = p.y + Y[i];
                    if (check(x, y)) {
                        marked[x][y] = true;
                        if (p.val > heightMap[x][y]) {
                            ans += (p.val - heightMap[x][y]);
                            // System.out.println(x + "," + y + "=" + (p.val - heightMap[x][y]));
                            heightMap[x][y] = p.val;
                        }
                        // 主要是为了确保短柱可以及时处理，至于为什么不一次性遍历全部是因为每个柱只能确保和它关联的区域的接水情况
                        priorityQueue.add(new Pair(heightMap[x][y], x, y));
                    }
                }
            }
        }
        return ans;
    }

    private boolean check(int x, int y) {
        return x >= 0 && y >= 0 && x < m && y < n && !marked[x][y];
    }

    private static class Pair {
        int val;
        int x, y;

        public Pair() {
        }

        public Pair(int val, int x, int y) {
            this.val = val;
            this.x = x;
            this.y = y;
        }

        @Override
        public String toString() {
            return "Pair{" +
                    "val=" + val +
                    ", x=" + x +
                    ", y=" + y +
                    '}';
        }
    }
}
```

#### [85. 最大矩形](https://leetcode-cn.com/problems/maximal-rectangle/)

题目：

给定一个仅包含 `0` 和 `1` 、大小为 `rows x cols` 的二维二进制矩阵，找出只包含 `1` 的最大矩形，并返回其面积。

 

**示例 1：**

![img](https://assets.leetcode.com/uploads/2020/09/14/maximal.jpg)



解答：

终于回到我们熟悉的单调栈了，这题关键在于怎么想到把二维的图形压缩成一维图形。其实前面有一题，是求直方图中**最大矩阵面积**，我们可以想办法靠上去，所以会分层计算，首先以第一层为底，进行计算，至于说矩阵的高度，就是连续的1的长度，如果期间出现了0，说明发生了断层，前面的高度将不被囊括，此时高度清0，重新开始即可。



此时我们有代码：

``` Java
class Solution {
    public int maximalRectangle(char[][] matrix) {
        if (matrix.length == 0 || matrix[0].length == 0) {
            return 0;
        }
        int[] tmp = new int[matrix[0].length];
        int ans = 0;
        // 二维压扁成一维，这是最关键的地方，也是我没想到的地方
        for (int i = 0; i < matrix.length; ++ i) {
            for (int j = 0; j < matrix[0].length; ++ j) {
                if (matrix[i][j] == '1') {
                    tmp[j] ++;
                } else {
                    tmp[j] = 0;
                }
            }
            ans = Math.max(ans, largestRectangleArea(tmp));
        }
        return ans;
    }

    private final LinkedList<Pair> stack = new LinkedList<>();

    private void add1(Pair pair) {
        if (stack.isEmpty()) {
            stack.add(pair);
        } else {
            Pair first;
            while (!stack.isEmpty() && (first = stack.getFirst()) != null && pair.val < first.val) {
                first.next = pair.index;
                stack.removeFirst();
            }
            stack.addFirst(pair);
        }
    }

    private void add2(Pair pair) {
        if (stack.isEmpty()) {
            stack.add(pair);
        } else {
            Pair first;
            while (!stack.isEmpty() && (first = stack.getFirst()) != null && pair.val < first.val) {
                first.prev = pair.index;
                stack.removeFirst();
            }
            stack.addFirst(pair);
        }
    }

    private int largestRectangleArea(int[] heights) {
        int ans = 0;
        Pair[] pairs = new Pair[heights.length];
        for (int i = 0; i < heights.length; ++ i) {
            Pair pair = new Pair();
            pair.val = heights[i];
            pair.index = i;
            pair.prev = -1;
            pair.next = pairs.length;
            pairs[i] = pair;
            add1(pair);
        }
        stack.clear();
        for (int i = heights.length-1; i >= 0; -- i) {
            add2(pairs[i]);
        }
        for (int i = 0; i < pairs.length; ++ i) {
            // System.out.println(pairs[i]);
            ans = Math.max(ans, pairs[i].val * (pairs[i].next - pairs[i].prev - 1));
        }
        return ans;
    }

    private static class Pair {
        int val, index;
        // 找到后面小于它的，和前面小于它的
        int prev, next;

        @Override
        public String toString() {
            return "val: " + val + ",index: " + index + ", prev: " + prev + ", next: " + next;
        }
    }
}
```

#### [44. 通配符匹配](https://leetcode-cn.com/problems/wildcard-matching/)

题目：

给定一个字符串 (s) 和一个字符模式 (p) ，实现一个支持 '?' 和 '*' 的通配符匹配。

'?' 可以匹配任何单个字符。
'*' 可以匹配任意字符串（包括空字符串）。
两个字符串完全匹配才算匹配成功。

说明:

s 可能为空，且只包含从 a-z 的小写字母。
p 可能为空，且只包含从 a-z 的小写字母，以及字符 ? 和 *。



解答：

又是通配符，又是两个字符串，直觉上DP。其实一般来说，**涉及到字符串匹配**的，包括回文串，上升子序列，上升子串，字符串匹配等，**基本都是DP解决**。



当然，像这道题，它就是硬编码，if-else硬匹配，枚举所有可能情况，其实涉及到匹配的基本都是这种套路，代码：

``` Java
class Solution {

    // 就硬做，没什么特殊的解法，枚举所有可能的状态转移方程
    public boolean isMatch(String s, String p) {
        char[] str1 = s.toCharArray();
        char[] str2 = p.toCharArray();
        boolean[][] dp = new boolean[str1.length+1][str2.length+1];
        if (str1.length == 0 && str2.length == 0) {
            return true;
        } else if (str2.length == 0) {
            return false;
        } else if (str1.length == 0) {
            for (char c : str2) {
                if (c != '*') {
                    return false;
                }
            }
            return true;
        }
        dp[0][0] = true;
        if (str2[0] == '*') {
            for (int i = 0; i < str1.length; ++ i) {
                dp[i+1][0] = true;
            }
        }
        for (int i = 0; i < str2.length; ++ i) {
            if (str2[i] == '*') {
                dp[0][i+1] = true;
            } else {
                break;
            }
        }
        for (int i = 1; i <= str1.length; ++ i) {
            for (int j = 1; j <= str2.length; ++ j) {
                if (str1[i-1] == str2[j-1]) {
                    dp[i][j] = dp[i-1][j-1];
                } else if (str2[j-1] == '?') {
                    dp[i][j] = dp[i-1][j-1];
                } else if (str2[j-1] == '*') {
                    for (int k = i; k >= 0; -- k) {
                        dp[i][j] = dp[i][j] || dp[k][j-1];
                        if (dp[i][j]) {
                            break;
                        }
                    }
                } else {
                    dp[i][j] = false;
                }
            }
        }
        return dp[str1.length][str2.length];
    }
}
```

#### [135. 分发糖果](https://leetcode-cn.com/problems/candy/)

题目：

老师想给孩子们分发糖果，有 N 个孩子站成了一条直线，老师会根据每个孩子的表现，预先给他们评分。

你需要按照以下要求，帮助老师给这些孩子分发糖果：

每个孩子至少分配到 1 个糖果。
评分更高的孩子必须比他两侧的邻位孩子获得更多的糖果。
那么这样下来，老师至少需要准备多少颗糖果呢？



解答：

就硬模拟：

``` Java
class Solution {

    // 就硬做，没有什么特殊的解法，两个循环跑一遍，来进行数量的更新
    public int candy(int[] ratings) {
        int[] vals = new int[ratings.length];
        vals[0] = 1;
        for (int i = 1; i < ratings.length; ++i) {
            if (ratings[i] > ratings[i - 1]) {
                vals[i] = vals[i - 1] + 1;
            } else {
                vals[i] = 1;
            }
        }
        for (int i = ratings.length - 2; i >= 0; --i) {
            if (ratings[i] > ratings[i + 1]) {
                // 唯一需要注意的是这里，第二遍循环从后往前时不能单纯的+1，还要考虑第一遍的更新结果不能被覆盖
                vals[i] = Math.max(vals[i], vals[i + 1] + 1);
            }
        }
        int ans = 0;
        for (int a : vals) {
            ans += a;
            // System.out.print(a + " ");
        }
        // System.out.println();
        return ans;
    }
}
```

#### [149. 直线上最多的点数](https://leetcode-cn.com/problems/max-points-on-a-line/)

题目：

给你一个数组 `points` ，其中 `points[i] = [xi, yi]` 表示 **X-Y** 平面上的一个点。求最多有多少个点在同一条直线上。



解答：

就硬循环，之所以提这个，是因为我当初以为暴力一定超时，现在我拎出来说就是想说，有些题需要暴力模拟：

``` Java
class Solution {
    public int maxPoints(int[][] points) {
        int ans = 1;
        for (int i = 0; i < points.length; ++ i) {
            for (int j = i+1; j < points.length; ++ j) {
                int a = points[j][0] - points[i][0];
                int b = points[j][1] - points[i][1];
                int count = 2;
                for (int k = j+1; k < points.length; ++ k) {
                    int c = points[k][0] - points[i][0];
                    int d = points[k][1] - points[i][1];
                    if (a*d == b*c) {
                        ++ count;
                    }
                }
                ans = Math.max(ans, count);
            }
        }
        return ans;
    }
}
```

#### [887. 鸡蛋掉落](https://leetcode-cn.com/problems/super-egg-drop/)

题目：

给你 k 枚相同的鸡蛋，并可以使用一栋从第 1 层到第 n 层共有 n 层楼的建筑。

已知存在楼层 f ，满足 0 <= f <= n ，任何从 高于 f 的楼层落下的鸡蛋都会碎，从 f 楼层或比它低的楼层落下的鸡蛋都不会破。

每次操作，你可以取一枚没有碎的鸡蛋并把它从任一楼层 x 扔下（满足 1 <= x <= n）。如果鸡蛋碎了，你就不能再次使用它。如果某枚鸡蛋扔下后没有摔碎，则可以在之后的操作中 重复使用 这枚鸡蛋。

请你计算并返回要确定 f 确切的值 的 最小操作次数 是多少？



解答：

这题可太艹了，为啥呢？这是谷歌当年的面试题，答出来的人并不多。首先我们知道，楼层越多，次数肯定越多，鸡蛋越少，次数肯定也越多，所以次数和楼层成正比，和鸡蛋数成反比。其实第一时间大家很容易想到的是二分查找，为了优化一波，我们可以使用DP，直接定义dp\[i]\[j]: 鸡蛋数为i时，楼层为j所需要的最少次数；在处理转移方程时，其实是需要遍历的，遍历已经求得的子区间，因为我们大概只能知道鸡蛋破碎的位置在中间或以下，甚至中间以上，所以我们需要来一次遍历，当上下关系变化时，就是我们要找的位置了。



其实如果想要优化，还可以使用二分替代遍历，此时可能就可以通过，但是当初我仅仅是想到查找这一步就想了许久，没有继续下去，我们来看另一种方法。



我们必须重新定义dp数组。这也是比较困难的动态规划常见的方式——无法直接定义。此时定义成dp\[i]\[j]: 当鸡蛋数为i时，尝试次数为j所能到达的最高高度。此时有dp\[i]\[j] = dp\[i-1]\[j-1] + dp\[i]\[j-1] + 1;分为碎了和没碎的总和。

``` Java
class Solution {
    public int superEggDrop(int k, int n) {
        // 下面来看一种题解里看到的比较难以想到的方法
        // 核心在于更改dp定义，这也是hard题里常见的：不要用要求来定义DP，而要用答案来定义DP。
        // 这里的dp为：dp[鸡蛋个数][操作次数]=楼层高，就是在有i个鸡蛋的情况下，操作j次，最高能操作到几层。
        // 经过测试最多最多14次
        int[][] dp = new int[k+1][n+1];
        int j = 0;
        while (dp[k][j] < n) {
            ++ j;
            for (int i = 1; i <= k; ++ i) {
                // 想要看当前操作+鸡蛋数能到多少层，就要看操作数-1能到的楼层数，此时还分为鸡蛋碎了，和鸡蛋没碎两种情况。
                dp[i][j] = dp[i-1][j-1] + dp[i][j-1] + 1;
            }
        }
        return j;
   
```

#### [115. 不同的子序列](https://leetcode-cn.com/problems/distinct-subsequences/)

题目：

给定一个字符串 s 和一个字符串 t ，计算在 s 的子序列中 t 出现的个数。

字符串的一个 子序列 是指，通过删除一些（也可以不删除）字符且不干扰剩余字符相对位置所组成的新字符串。（例如，"ACE" 是 "ABCDE" 的一个子序列，而 "AEC" 不是）

题目数据保证答案符合 32 位带符号整数范围。



解答：

典型的字符串DP问题，直接定义即可，dp\[i]\[j]: 表示t的钱i个串在s的前j个串中出现的次数。此时我们有dp\[i]\[j] = dp\[i-1]\[j-1] + dp\[i]\[j-1]，这是在t\[i] == s\[j]的情况下，或者dp\[i]\[j] = dp\[i]\[j-1]这是在不等情况下。

``` Java
class Solution {
    public int numDistinct(String s, String t) {
        s = " " + s;
        t = " " + t;
        char[] str1 = t.toCharArray();
        char[] str2 = s.toCharArray();
        int[][] dp = new int[str1.length+1][str2.length+1];
        dp[0][0] = 1;
        for (int i = 1; i <= str1.length; ++ i) {
            for (int j = 1; j <= str2.length; ++ j) {
                if (str1[i-1] == str2[j-1]) {
                    dp[i][j] = dp[i-1][j-1]+dp[i][j-1];
                } else {
                    dp[i][j] = dp[i][j-1];
                }
            }
        }
        return dp[str1.length][str2.length];
    }
}
```

#### 