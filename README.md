- Typescript
    

    
    - Array
        1. Reverse Array
        
        ```tsx
        let arr=[1,2,3,4,5];
        reverseArr(arr);
        console.log(arr);
        function reverseArr(arr:number[]){
            let left=0;
            let right=arr.length-1;
            while(left<right){
                let temp=arr[left];
                arr[left]=arr[right];
                arr[right]=temp;
                left++;
                right--;
            }
        }
        ```
        
        1. minMaxArray
        
        ```tsx
        const arr=[1,3,4,6,7,5];
        getMinMax(arr);
        function getMinMax(arr:number[]){
        let n=arr.length;
        secondLargest(arr);
        secondSmallest(arr);
        }
        function secondLargest(arr:number[]){
            let largest=-Infinity;
            let secondLargest=-Infinity;
            for(let i=0;i<arr.length;i++){
                if(arr[i]>largest){
                    secondLargest=largest;
                    largest=arr[i];
                }else if(arr[i]>=secondLargest && arr[i]!=largest){
                    secondLargest=arr[i];
                }
            }
            console.log("largest",largest);
            console.log("secondLargest",secondLargest);
        
        }
        function secondSmallest(arr:number[]){
            let smallest=Infinity;
            let secondSmallest=Infinity;
            for(let i=0;i<arr.length;i++){
                if(arr[i]<arr[smallest]){
                    secondSmallest=smallest;
                    smallest=arr[i];
                }else if(arr[i]<=secondSmallest && arr[i]!=smallest){
                    secondSmallest=arr[i];
                }
            }
             console.log("smallest", smallest);
          console.log("second smallest", secondSmallest);
        }
        ```
        
        1. Kth smallest
        
        ```tsx
        let arr = [1, 2, 3, 29, 344, 23];
        let k = 4;
        kSmallest(arr, k);
        
        function kSmallest(arr: number[], k: number): void {
            if (k > 0 && k <= arr.length) {
                arr.sort((a, b) => a - b);
                console.log(arr[k - 1]);
            } else {
                console.log("Invalid value of k");
            }
        }
        
        ```
        
        1. Sort (0,1,2)
        
        ```tsx
        let arr=[0,2,1,2,0];
        sortArray(arr);
        function swap(arr:number[],first:number,second:number):void{
            [arr[first],arr[second]]=[arr[second],arr[first]];
        }
        function sortArray(arr:number[]){
            let n=arr.length;
            let low=0,mid=0,high=n-1;
            while(mid<=high){
                if(arr[mid]==0){
                    swap(arr,low++,mid++);
                }else if(arr[mid]==2){
                    swap(arr,mid,high--);
                }else{
                    mid++;
                }
            }
            console.log(arr);
        }
        ```
        
        1. Move Negative left
        
        ```tsx
        let arr=[-1,2,3,-3,4,-5];
        sortArray(arr);
        function sortArray(arr:number[]){
            let n=arr.length;
            let j=0;
            for(let i=0;i<arr.length;i++){
                if(arr[i]<0){
                    if(i!=j){
                        let temp=arr[i];
                        arr[i]=arr[j];
                        arr[j]=temp;
                    }
                    j++;
                }
            }
            console.log(arr);
        }
        ```
        
        1. Find Union and InterSection
        
        ```tsx
        let arr1 = [1, 2, 3, 5, 6, 7, 8, 23];
        let arr2 = [2, 3, 4, 5, 6];
        const { union, intersection } = findUnionAndIntersection(arr1, arr2);
        
        console.log("Union:", union);
        console.log("Intersection:", intersection);
        
        function findUnionAndIntersection(arr1: number[], arr2: number[]): { union: number[], intersection: number[] } {
            const map = new Map<number, number>();
            for (const item of arr1) {
                map.set(item, (map.get(item) || 0) + 1);
            }
        
            const unionSet = new Set(arr1);
            const intersection: number[] = [];
        
            for (const item of arr2) {
                if (map.has(item) && map.get(item)! > 0) {
                    map.set(item, map.get(item)! - 1);
                    intersection.push(item);
                }
                unionSet.add(item);  // add item to the union set
            }
        
            return {
                union: Array.from(unionSet),  // convert the Set back to an array for the union
                intersection
            };
        }
        
        ```
        
        1. Cyclic Rotate Array
        
        ```tsx
        let arr=[1,2,3,4,4,6,8];
        let k=2;
        rotateLeft(arr,k);
        rotateRight(arr,k);
        function reverse(arr:number[],l:number,r:number):void{
            while(l<r){
                let temp=arr[l];
                arr[l]=arr[r];
                arr[r]=temp;
                l++;
                r--;
            }
        }
        function rotateLeft(arr:number[],k:number):void{
        let n=arr.length;
            reverse(arr,0,k-1);
            reverse(arr,k,n-1);
            reverse(arr,0,n-1);
            console.log("reverse left= ",arr);
        }
        function rotateRight(arr:number[], k:number):void {
          let n = arr.length;
          reverse(arr, 0, n - k - 1);
          reverse(arr, n - k, n - 1);
          reverse(arr, 0, n - 1);
          console.log("reverse right = ", arr);
        }
        
        ```
        
        1. Sub Array Kadane
        
        ```tsx
        const arr=[3,4,5,6,-1];
        const n=arr.length;
        const maxSum=maxSubArray(arr,n);
        console.log(`The maximum subarray sum is: ${maxSum}`);
        function maxSubArray(arr:number[],n:number):number{
            let max=Number.MIN_SAFE_INTEGER;
            let sum=0;
            for(let i=0;i<n;i++){
                sum+=arr[i];
                if(sum>max){
                    max=sum;
                }if(sum<0){
                    sum=0;
                }
            }
            return max;
        }
        ```
        
        1. Minimise Height
        
        ```tsx
        const arr=[1,3,4,5,67,7];
        const k=2;
        console.log(getMinDiff(arr,arr.length,k));
        function getMinDiff(arr:number[], n:number, k:number):number{
        if(n==1){
            return 0;
        }
        arr.sort((a,b)=>a-b);
        let ans=arr[n-1]-arr[0];
        let small=arr[0]+k;
        let big=arr[n-1]-k;
        if(small>big){
            [small,big]=[big,small];
        }
        for (let i = 1; i < n - 1; i++) {
            const height = arr[i];
            const subtract = height - k;
            const add = height + k;
        
            if (subtract >= small || add <= big) {
              continue;
            }
        
            if (big - subtract <= add - small) {
              small = subtract;
            } else {
              big = add;
            }
          }
        
          return Math.min(ans, big - small);
        }
        
        ```
        
        1. Minimum Number of Jump
        
        ```tsx
        let arr = [2, 3, 4, 5, 2, 3, 4, 8, 12];
        let n = arr.length;
        console.log("Minimum Jumps: ", minJumpBruteForce(arr, n));
        
        function minJumpBruteForce(arr: number[], n: number): number {
            // If array has only one element, no jumps are needed
            if (n === 1) return 0;
            
            // If the first element is 0, we can't make any jump
            if (arr[0] === 0) return -1;
            
            let rng = arr[0];  // Maximum range reachable with current jump
            let sl = arr[0];   // Steps left in the current jump
            let jp = 1;        // Number of jumps taken
        
            for (let i = 1; i < n; i++) {
                // Check if we've reached the end
                if (i === n - 1) return jp;
                
                // Update the maximum range reachable
                rng = Math.max(rng, i + arr[i]);
                sl--;  // Decrease steps left in the current jump range
        
                // If steps left become zero, we need another jump
                if (sl === 0) {
                    jp++;  // Increase jump count
        
                    // If the current maximum range is less than or equal to `i`, we can't move forward
                    if (rng <= i) return -1;
        
                    // Reset the steps left to reach `rng` from current position `i`
                    sl = rng - i;
                }
            }
        
            // If we never reach the last index, return -1
            return -1;
        }
        
        ```
        
        1. Find Duplicates
        
        ```tsx
        let arr = [1, 3, 4, 2, 3]; // Corrected array where 3 is the duplicate.
        console.log("Duplicate no: ", findDuplicate(arr));
        
        function findDuplicate(arr: number[]): number {
            let slow = arr[0];
            let fast = arr[0];
            
            // Phase 1: Detect cycle using slow and fast pointers
            do {
                slow = arr[slow];         // move slow by 1 step
                fast = arr[arr[fast]];    // move fast by 2 steps
            } while (slow !== fast);      // continue until they meet
        
            // Phase 2: Find the entry point of the cycle (duplicate number)
            fast = arr[0];  // Reset fast to the beginning of the array
            while (slow !== fast) {
                slow = arr[slow];   // move both slow and fast by 1 step
                fast = arr[fast];
            }
            
            return slow;  // Both pointers now point to the duplicate number
        }
        
        ```
        
        1. Merge 
        
        ```tsx
        let arr1=[1,4,8,10];
        let arr2=[2,3,4];
        let n=4;
        let m=3;
        merge(arr1,arr2,n,m);
        console.log("The merged arrays are: ");
        console.log("arr1[] = " + arr1.join(" "));
        console.log("arr2[] = " + arr2.join(" "));
        function merge(arr1:number[],arr2:number[],n:number,m:number):void{
            let left=n-1;
            let right=0;
            while(left>0 && right<m){
                if(arr1[left]>arr2[right]){
                    [arr1[left],arr2[right]]=[arr2[right],arr1[left]];
                    left--;
                    right++;
                }else{
                    break;
                }
            }
            arr1.sort((a, b) => a - b);
          arr2.sort((a, b) => a - b);
        }
        ```
        
        1. MergeIntervals
        
        ```tsx
        
        const arr = [
          [1, 3],
          [8, 10],
          [2, 6],
          [15, 18],
        ];
        const ans = mergeOverlappingInterval(arr);
        console.log("The merged intervals are:");
        for (let it of ans) {
          console.log(`[${it[0]}, ${it[1]}]`);
        }
        
        function mergeOverlappingInterval(arr: number[][]): number[][] {
          let n = arr.length;
          arr.sort((a, b) => a[0] - b[0]);  // Sort intervals by the starting point
        
          const ans: number[][] = [arr[0]];  // Initialize the result with the first interval
          for (let i = 1; i < n; i++) {
            const last = ans[ans.length - 1];  // Get the last interval in the result
            const curr = arr[i];  // Get the current interval
        
            // Check if there is an overlap
            if (curr[0] <= last[1]) {
              // Merge the intervals by updating the end of the last interval
              last[1] = Math.max(curr[1], last[1]);
            } else {
              // No overlap, so add the current interval to the result
              ans.push(curr);
            }
          }
        
          return ans;  // Return the merged intervals
        }
        
        ```
        
        1. Next Permutation
        
        ```tsx
        let a = [2, 4, 5, 6, 77, 6];
        let ans = nextGreaterPermutation(a);
        console.log("The next permutation is: [" + ans.join(" ") + "]");
        
        function nextGreaterPermutation(arr: number[]): number[] {
            let n = arr.length;
            let index = -1;
        
            // Step 1: Find the first index where arr[i] < arr[i+1]
            for (let i = n - 2; i >= 0; i--) {
                if (arr[i] < arr[i + 1]) {
                    index = i;
                    break;
                }
            }
        
            // Step 2: If no such index is found, reverse the array
            if (index === -1) {
                arr.reverse();
                return arr;
            }
        
            // Step 3: Find the smallest number greater than arr[index] from the right side
            for (let i = n - 1; i > index; i--) {
                if (arr[i] > arr[index]) {
                    [arr[i], arr[index]] = [arr[index], arr[i]]; // Swap them
                    break;
                }
            }
        
            // Step 4: Reverse the subarray to the right of index
            arr.splice(index + 1, n - index - 1, ...arr.slice(index + 1).reverse());
        
            return arr;
        }
        
        ```
        
        1. Count Inversion
        
        ```tsx
        const a: number[] = [5, 4, 3, 2, 1];
        const cnt: number = numberOfInversionsOptimal(a);
        console.log("The number of inversions is: " + cnt);
        
        function numberOfInversionsNaive(arr: number[]): number {
          let count = 0;
          for (let i = 0; i < arr.length; i++) {
            for (let j = i + 1; j < arr.length; j++) { // Changed 'i' to 'i+1' for proper comparison
              if (arr[i] > arr[j]) {
                count++;
              }
            }
          }
          return count;
        }
        
        function mergeSort(arr: number[], low: number, high: number): number {
          let count = 0;
          if (low >= high) return count;
          let mid = Math.floor((low + high) / 2);
          count += mergeSort(arr, low, mid);
          count += mergeSort(arr, mid + 1, high);
          count += merge(arr, low, mid, high);
          return count;
        }
        
        function merge(arr: number[], low: number, mid: number, high: number): number {
          const temp: number[] = [];
          let left = low;
          let right = mid + 1;
          let count = 0;
        
          while (left <= mid && right <= high) {
            if (arr[left] <= arr[right]) {
              temp.push(arr[left]);
              left++;
            } else {
              temp.push(arr[right]);
              count += mid - left + 1;
              right++;
            }
          }
        
          while (left <= mid) {
            temp.push(arr[left]);
            left++;
          }
        
          while (right <= high) {
            temp.push(arr[right]);
            right++;
          }
        
          for (let i = low; i <= high; i++) {
            arr[i] = temp[i - low];
          }
        
          return count;
        }
        
        function numberOfInversionsOptimal(arr: number[]): number {
          return mergeSort(arr, 0, arr.length - 1);
        }
        
        ```
        
        1. Stock Buy And Sell
        
        ```tsx
        let arr=[7,1,4,6,8,2];
        let n=arr.length;
        let maxProfit=getMaxProfit(arr,n);
        console.log("Max Profit: ", maxProfit);
        function getMaxProfit(arr:number[], n:number):number{
            let ans=0;
            let minPrice=Infinity;
            for(let i=0;i<n;i++){
                minPrice=Math.min(minPrice,arr[i]);
                ans=Math.max(ans,arr[i]-minPrice);
            }
            return ans;
        }
        ```
        
        17.  Compare Pair Sum
        
        ```tsx
        let arr: number[] = [1, 5, 7, 1];
        let k: number = 6;
        let n: number = arr.length;
        console.log("The number of pairs with sum:", k, ":", countPairs(arr, n, k));
        
        function countPairs(arr: number[], n: number, k: number): number {
            let count = 0;
            const freqMap = new Map<number, number>(); // Explicitly specifying the type of the map
            for (let i = 0; i < n; i++) {
                const required = k - arr[i];
                if (freqMap.has(required)) {
                    count += freqMap.get(required)!; // Use non-null assertion since we know the value exists
                }
                freqMap.set(arr[i], (freqMap.get(arr[i]) || 0) + 1);
            }
            return count;
        }
        
        ```
        
        1. Common Elements 3 sorted Array
        
        ```tsx
        let A: number[] = [1, 5, 10, 20, 40, 80],
            n1: number = A.length;
        let B: number[] = [6, 7, 20, 80, 100],
            n2: number = B.length;
        let C: number[] = [3, 4, 15, 20, 30, 70, 80, 120],
            n3: number = C.length;
        
        console.log("ans: ", commonElements(A, B, C, n1, n2, n3));
        
        // TC: O(n1 + n2 + n3) SC: O(1)
        function commonElements(arr1: number[], arr2: number[], arr3: number[], n1: number, n2: number, n3: number): number[] {
          let i = 0, j = 0, k = 0;
          let res: number[] = [];
          let last: number = Number.MIN_SAFE_INTEGER;
          
          while (i < n1 && j < n2 && k < n3) {
            if (arr1[i] === arr2[j] && arr1[i] === arr3[k] && arr1[i] !== last) {
              res.push(arr1[i]);
              last = arr1[i];
              i++;
              j++;
              k++;
            } else if (Math.min(arr1[i], arr2[j], arr3[k]) === arr1[i]) {
              i++;
            } else if (Math.min(arr1[i], arr2[j], arr3[k]) === arr2[j]) {
              j++;
            } else {
              k++;
            }
          }
          
          if (res.length === 0) {
            return [-1];
          }
          
          return res;
        }
        
        ```
        
        1. Rearrange by Sign
        
        ```tsx
        let a: number[] = [1, -2, 3, -4, 5, -6];
        let n: number = a.length;
        let ans = rearrangeBySign(a, n);
        console.log(ans.join(" "));
        
        function rearrangeBySign(a: number[], n: number): number[] {
            let ans = new Array(n);
            let posIndex = 0;
            let negIndex = 1;
        
            // First, place negative numbers at odd indices and positive numbers at even indices
            for (let i = 0; i < n; i++) {
                if (a[i] < 0) {
                    ans[negIndex] = a[i];
                    negIndex += 2; // Move to next odd index
                } else {
                    ans[posIndex] = a[i];
                    posIndex += 2; // Move to next even index
                }
            }
        
            // In case there are more positives or negatives and you still have unfilled spots, fill them in the remaining spots
            if (posIndex < n) {
                for (let i = 0; i < n; i++) {
                    if (ans[i] === undefined) {
                        ans[i] = a[i];
                    }
                }
            }
            
            return ans;
        }
        
        ```
        
        1. Sub Array k0
        
        ```tsx
        let arr: number[] = [4, 2, -3, 1, 6];
        let n: number = arr.length;
        let k: number = 0;
        
        console.log("Subarray with sum k:", findSubarrayOptimal(arr, n, k));
        
        // TC: O(n^2) - Naive solution
        function findSubarrayNaive(arr: number[], n: number, k: number): boolean {
          let isPresent = false;
        
          for (let i = 0; i < n; i++) {
            let sum = 0;
            for (let j = i; j < n; j++) {
              sum += arr[j];
              if (sum === k) {
                isPresent = true;
                break;
              }
            }
            if (isPresent) break;
          }
          return isPresent;
        }
        
        // TC: O(n) - Optimal solution using prefix sum and hash set
        function findSubarrayOptimal(arr: number[], n: number, k: number): boolean {
          let sum = 0;
          let isPresent = false;
          let prefixSum = new Set<number>();
        
          for (let i = 0; i < n; i++) {
            sum += arr[i];
        
            // Check if the current sum is equal to k, or if the difference (sum - k) exists in the prefixSum set
            if (sum === k || prefixSum.has(sum - k)) {
              isPresent = true;
              break;
            }
        
            // Add the current sum to the prefixSum set for future checks
            prefixSum.add(sum);
          }
        
          return isPresent;
        }
        
        ```
        
        1. Max Product Sub Array
        
        ```tsx
        let arr: number[] = [6, -3, -10, 0, 2];
        let n: number = arr.length;
        console.log("Max Product: ", getMaxProduct(arr, n));
        
        function getMaxProduct(arr: number[], n: number): number {
            let maxEnd = arr[0];
            let minEnd = arr[0];
            let maxProd = arr[0];
        
            for (let i = 1; i < n; i++) {
                if (arr[i] < 0) {
                    // Swap maxEnd and minEnd when encountering a negative number
                    let temp = maxEnd;
                    maxEnd = minEnd;
                    minEnd = temp;
                }
        
                // Calculate max and min products ending at the current position
                maxEnd = Math.max(arr[i], maxEnd * arr[i]);
                minEnd = Math.min(arr[i], minEnd * arr[i]);
        
                // Update maxProd with the maximum product found so far
                maxProd = Math.max(maxProd, maxEnd);
            }
        
            return maxProd;
        }
        
        ```
        
        1. Longes Consecutive Sucessive
        
        ```tsx
        let arr: number[] = [100, 200, 1, 3, 4];
        let ans = longestSuccessive(arr);
        console.log("The longest consecutive sequence is:", ans);
        
        function longestSuccessive(arr: number[]): number {
            let n = arr.length;
            if (n === 0) return 0;
        
            let longest = 1;
            let st = new Set(arr);
        
            for (let it of st) {
                // Only start counting sequence if it's the beginning of a sequence
                if (!st.has(it - 1)) {
                    let cnt = 1;
                    let x = it;
                    while (st.has(x + 1)) {
                        x += 1;
                        cnt += 1;
                    }
                    longest = Math.max(longest, cnt);
                }
            }
            return longest;
        }
        
        ```
        
        1. Element Appear NK times
        
        ```tsx
        let arr: number[] = [2, 2, 1, 1, 1, 2, 2];
        let n: number = arr.length;
        let k: number = 2;
        
        // Naive Approach: Check each element and count its frequency
        function majorityElementNaive(arr: number[], n: number, k: number): number {
          for (let i = 0; i < n; i++) {
            let cnt = 0;
            for (let j = 0; j < n; j++) {
              if (arr[i] == arr[j]) {
                cnt++;
              }
            }
            if (cnt > Math.floor(n / k)) {
              return arr[i];
            }
          }
          return -1;
        }
        
        // Better Approach: Use a hashmap to store frequencies
        function majorityElementBetter(arr: number[], n: number, k: number): number[] {
          let result: number[] = [];
          const freqMap = new Map<number, number>();
          const threshold = Math.floor(n / k) + 1;
        
          for (let i = 0; i < n; i++) {
            freqMap.set(arr[i], (freqMap.get(arr[i]) || 0) + 1);
        
            // If frequency meets threshold, add to result and remove from map to prevent duplicates
            if (freqMap.get(arr[i]) === threshold) {
              result.push(arr[i]);
              freqMap.delete(arr[i]);
            }
          }
          return result;
        }
        
        // Majority Element for n/2 using Boyer-Moore Voting Algorithm
        function majorityElementHalf(arr: number[]): number {
          let count = 0;
          let candidate: number | undefined;
        
          for (let num of arr) {
            if (count === 0) {
              candidate = num;
            }
            count += (num === candidate) ? 1 : -1;
          }
        
          // Verify candidate frequency
          let verifyCount = arr.filter(num => num === candidate).length;
          return verifyCount > Math.floor(arr.length / 2) ? candidate! : -1;
        }
        
        // Majority Elements for n/3 using Extended Boyer-Moore Voting Algorithm
        function majorityElementThird(arr: number[]): number[] {
          let n = arr.length;
          let count1 = 0, count2 = 0;
          let candidate1: number | undefined, candidate2: number | undefined;
        
          for (let num of arr) {
            if (candidate1 === num) {
              count1++;
            } else if (candidate2 === num) {
              count2++;
            } else if (count1 === 0) {
              candidate1 = num;
              count1 = 1;
            } else if (count2 === 0) {
              candidate2 = num;
              count2 = 1;
            } else {
              count1--;
              count2--;
            }
          }
        
          // Verify candidates
          count1 = arr.filter(num => num === candidate1).length;
          count2 = arr.filter(num => num === candidate2).length;
          let result: number[] = [];
          let threshold = Math.floor(n / 3) + 1;
        
          if (count1 >= threshold) result.push(candidate1!);
          if (count2 >= threshold) result.push(candidate2!);
        
          return result;
        }
        
        // Testing the functions
        console.log("Naive Approach Majority Element:", majorityElementNaive(arr, n, k));
        console.log("Better Approach Majority Element:", majorityElementBetter(arr, n, k));
        console.log("Boyer-Moore Voting (n/2 Majority):", majorityElementHalf(arr));
        console.log("Extended Boyer-Moore Voting (n/3 Majority):", majorityElementThird(arr));
        
        ```
        
        1. Max Profit 2 Data
        
        ```tsx
        let price: number[] = [2, 30, 15, 10, 8, 25, 80];
        let n: number = price.length;
        console.log("Maximum Profit = ", maxProfitOptimal(price, n));
        
        // Function to calculate maximum profit with two transactions (O(2n) approach)
        function maxProfit(arr: number[], n: number): number {
          let profit: number[] = Array(n).fill(0);
          let maxPrice = arr[n - 1];
        
          // Traverse from the right to find maximum selling profit
          for (let i = n - 2; i >= 0; i--) {
            if (arr[i] > maxPrice) {
              maxPrice = arr[i];
            }
            profit[i] = Math.max(profit[i + 1], maxPrice - arr[i]);
          }
        
          let minPrice = arr[0];
        
          // Traverse from the left to find maximum buying profit
          for (let i = 1; i < n; i++) {
            if (arr[i] < minPrice) {
              minPrice = arr[i];
            }
            profit[i] = Math.max(profit[i - 1], profit[i] + (arr[i] - minPrice));
          }
        
          return profit[n - 1];
        }
        
        // Optimized function to calculate maximum profit with two transactions (O(n) approach)
        function maxProfitOptimal(arr: number[], n: number): number {
          let first_buy = -Infinity;
          let first_sell = 0;
          let second_buy = -Infinity;
          let second_sell = 0;
        
          for (let i = 0; i < n; i++) {
            first_buy = Math.max(first_buy, -arr[i]); // max profit after buying the first stock
            first_sell = Math.max(first_sell, first_buy + arr[i]); // max profit after selling the first stock
            second_buy = Math.max(second_buy, first_sell - arr[i]); // max profit after buying the second stock
            second_sell = Math.max(second_sell, second_buy + arr[i]); // max profit after selling the second stock
          }
        
          return second_sell; // maximum profit with at most two transactions
        }
        
        ```
        
        1. Subset of Other Array
        
        ```tsx
        let a1: number[] = [11, 7, 1, 13, 21, 3, 7, 3];
        let n: number = a1.length;
        let a2: number[] = [11, 3, 7, 1, 7];
        let m: number = a2.length;
        console.log(isSubsetOptimal(a1, a2, n, m));
        
        // Naive approach to check if `a2` is a subset of `a1` (TC: O(n^2))
        function isSubsetNaive(a1: number[], a2: number[], n: number, m: number): string {
          let count = 0;
        
          for (let i = 0; i < m; i++) {
            let found = false;
            for (let j = 0; j < n; j++) {
              if (a1[j] === a2[i]) {
                count++;
                a1[j] = -1; // Mark as used
                found = true;
                break;
              }
            }
            if (!found) {
              return "No"; // Element from a2 not found in a1
            }
          }
        
          return m === count ? "Yes" : "No";
        }
        
        // Optimal approach using a hashmap to check if `a2` is a subset of `a1` (TC: O(n + m))
        function isSubsetOptimal(a1: number[], a2: number[], n: number, m: number): string {
          let freq = new Map<number, number>();
        
          // Count frequencies of elements in a1
          for (let i = 0; i < n; i++) {
            freq.set(a1[i], (freq.get(a1[i]) || 0) + 1);
          }
        
          // Check elements of a2 in the frequency map
          for (let num of a2) {
            if (!freq.has(num)) {
              return "No";
            }
            freq.set(num, freq.get(num)! - 1);
            if (freq.get(num)! < 0) {
              return "No";
            }
          }
        
          return "Yes";
        }
        
        ```
        
        1. Three Sum
        
        ```tsx
        let arr: number[] = [-1, 0, 1, 2, -1, -4];
        let n: number = arr.length;
        let k: number = 0;
        let res = tripletOptimal(n, arr, k);
        console.log("triplets: ", res);
        
        // Optimal approach (TC: O(n log n) + O(n^2), SC: O(no. of triplets))
        function tripletOptimal(n: number, arr: number[], val: number): number[][] {
          let ans: number[][] = [];
          arr.sort((a, b) => a - b);
        
          for (let i = 0; i < n; i++) {
            if (i !== 0 && arr[i] === arr[i - 1]) continue;
        
            let j = i + 1;
            let k = n - 1;
            while (j < k) {
              let sum = arr[i] + arr[j] + arr[k];
              if (sum < val) {
                j++;
              } else if (sum > val) {
                k--;
              } else {
                ans.push([arr[i], arr[j], arr[k]]);
                j++;
                k--;
        
                // Skip duplicates for the second and third elements
                while (j < k && arr[j] === arr[j - 1]) j++;
                while (j < k && arr[k] === arr[k + 1]) k--;
              }
            }
          }
        
          return ans;
        }
        
        ```
        
        1. Trapping RainWater
        
        ```tsx
        function trappingRainwater(height: number[]): number {
          if (height.length === 0) return 0;
        
          let left = 0;
          let right = height.length - 1;
          let leftMax = 0;
          let rightMax = 0;
          let waterTrapped = 0;
        
          while (left <= right) {
            if (height[left] <= height[right]) {
              if (height[left] >= leftMax) {
                leftMax = height[left];
              } else {
                waterTrapped += leftMax - height[left];
              }
              left++;
            } else {
              if (height[right] >= rightMax) {
                rightMax = height[right];
              } else {
                waterTrapped += rightMax - height[right];
              }
              right--;
            }
          }
        
          return waterTrapped;
        }
        
        // Example usage:
        const height = [0, 1, 0, 2, 1, 0, 1, 3, 2, 1, 2, 1];
        console.log("Trapped Rainwater:", trappingRainwater(height));
        
        ```
        
        1. Factorial of large number
        
        ```tsx
        function largeNumberFactorial(n: number): number[] {
          let result: number[] = [1]; // Initialize result to handle large numbers
        
          for (let i = 2; i <= n; i++) {
            multiply(i, result);
          }
        
          return result.reverse(); // Reverse the array for readability
        }
        
        function multiply(x: number, result: number[]): void {
          let carry = 0;
          
          for (let i = 0; i < result.length; i++) {
            let product = result[i] * x + carry;
            result[i] = product % 10;
            carry = Math.floor(product / 10);
          }
        
          while (carry > 0) {
            result.push(carry % 10);
            carry = Math.floor(carry / 10);
          }
        }
        
        // Example usage:
        const n = 100;
        console.log(`Factorial of ${n}:`, largeNumberFactorial(n).join(""));
        
        ```
        
    - Binary Search
        1. Find X in Array
        
        ```tsx
        let arr: number[] = [3, 4, 6, 7, 9, 12, 16];
        let target: number = 6;
        let index = search(arr, target);
        if (index == -1) {
            console.log("The target is not present");
        } else {
            console.log("The target is at index:", index);
        }
        
        function search(arr: number[], target: number): number {
            return binarySearch(arr, target);
        }
        
        function binarySearch(arr: number[], target: number): number {
            let low = 0;
            let high = arr.length - 1; // Set high to the last valid index
        
            while (low <= high) { // Adjust condition to <=
                let mid = Math.floor((low + high) / 2);
                if (arr[mid] === target) {
                    return mid;
                } else if (arr[mid] < target) {
                    low = mid + 1;
                } else {
                    high = mid - 1;
                }
            }
            return -1;
        }
        
        ```
        
        1. Lower Bound
        
        ```tsx
        let arr: number[] = [3, 5, 9, 12];
        let x: number = 9;
        let index = lowerBound(arr, arr.length, x);
        console.log("The lower bound is at index:", index);
        
        function lowerBound(arr: number[], n: number, x: number): number {
            let low = 0, high = n - 1;
            let ans = n; // Initialize ans with n, indicating "not found" if all elements are less than x
        
            while (low <= high) {
                let mid = Math.floor((low + high) / 2);
                if (arr[mid] >= x) {
                    ans = mid; // Update ans with the current mid index
                    high = mid - 1; // Move left to find the lowest index
                } else {
                    low = mid + 1; // Move right if arr[mid] < x
                }
            }
            return ans;
        }
        
        ```
        
        1. Upper Bound
        
        ```tsx
        let arr: number[] = [3, 5, 8, 9, 15, 19];
        let x: number = 9;
        let ind = upperBoundOptimal(arr, arr.length, x);
        console.log("The upper bound is at index:", ind);
        
        // TC: O(log n)
        function upperBoundOptimal(arr: number[], n: number, x: number): number {
          let low = 0;
          let high = n - 1;
          let ans = n; // Initialize ans with n, indicating "not found" if all elements are less than or equal to x
        
          while (low <= high) {
            let mid = Math.floor((low + high) / 2);
            if (arr[mid] > x) {
              ans = mid; // Update ans with the current mid index
              high = mid - 1; // Move left to find the lowest index with arr[mid] > x
            } else {
              low = mid + 1; // Move right if arr[mid] <= x
            }
          }
          return ans;
        }
        
        ```
        
        1. Search Insert Position
        
        ```tsx
        let arr:number[]=[1,3,4,7];
        let x=6;
        let index=searchInsert(arr,x);
        console.log("The index is: ",index);
        function searchInsert(arr:number[], x:number):number{
            let n=arr.length;
            let low=0;
            let high=n-1;
            let ans=n;
            while(low<=high){
                let mid=Math.floor((low+high)/2);
                if(arr[mid]>=x){
                    ans=mid;
                    high=mid-1;
                }else{
                    low=mid+1;
                }
                
            }
            return ans;
        }
        ```
        
        1. Floor Ceil
        
        ```tsx
        let arr: number[] = [3, 4, 4, 7, 8, 10];
        let n: number = arr.length;
        let x: number = 8;
        let ans = getFloorAndCeil(arr, n, x);
        console.log("The floor and ceil are:", ans[0], ans[1]);
        
        function getFloorAndCeil(arr: number[], n: number, x: number): [number, number] {
          let floor = findFloor(arr, n, x);
          let ceil = findCeil(arr, n, x);
          return [floor, ceil];
        }
        
        function findFloor(arr: number[], n: number, x: number): number {
          let low = 0;
          let high = n - 1;
          let ans = -1;
        
          while (low <= high) {
            let mid = Math.floor((low + high) / 2);
            if (arr[mid] <= x) {
              ans = arr[mid];
              low = mid + 1;
            } else {
              high = mid - 1;
            }
          }
          return ans;
        }
        
        function findCeil(arr: number[], n: number, x: number): number {
          let low = 0;
          let high = n - 1;
          let ans = -1;
        
          while (low <= high) {
            let mid = Math.floor((low + high) / 2);
            if (arr[mid] >= x) {
              ans = arr[mid];
              high = mid - 1;
            } else {
              low = mid + 1;
            }
          }
          return ans;
        }
        
        ```
        
        1. First Last Occurance
        
        ```tsx
        let arr:number[]=[3,4,5,6,12,12,32];
        let n:number=arr.length;
        let k:number=12;
        let ans=getLastOccurance(arr,n,k);
        console.log("answer: ",ans);
        function getLastOccurance(arr:number[],n:number,k:number):number{
            let low=0,high=n-1,result=-1;
            while(low<=high){
                let mid=Math.floor(low+(high-low)/2);
                if(arr[mid]==k){
                    result=mid;
                    high=mid-1;
                }
                else if(k<arr[mid]){
                    high=mid-1;
                }else{
                    low=mid+1;
                }
            }
            return result;
        }
        ```
        
        1. Count Occurance
        
        ```tsx
        
        ```
        
        1. 
        
        1. Search In Rotate
    - Polly Fills
        1. Filter
        
        ```tsx
        interface Array<T> {
          myOwnFilter(callback: (element: T, index: number) => boolean): T[];
        }
        
        Array.prototype.myOwnFilter = function <T>(callback: (element: T, index: number) => boolean): T[] {
          let newArr: T[] = [];
          this.forEach((element: T, index: number) => {
            if (callback(element, index)) {
              newArr.push(element);
            }
          });
          return newArr;
        };
        
        let arr: number[] = [1, 3, 2, 4, 9, 5, 8, 6];
        
        const arrFiltered = arr.myOwnFilter((element, index) => {
          return element > 4;
        });
        
        console.log(arrFiltered);
        
        ```
        
        1. For Each
        
        ```tsx
        interface Array<T> {
          myForEach(callback: (element: T, index: number) => void): void;
        }
        
        Array.prototype.myForEach = function <T>(callback: (element: T, index: number) => void): void {
          for (let i = 0; i < this.length; i++) {
            callback(this[i], i);
          }
        };
        
        const arr: number[] = [1, 2, 3, 4, 5];
        
        arr.myForEach((element, index) => {
          console.log(`Element at index ${index}: ${element}`);
        });
        
        ```
        
        1. Map
        
        ```tsx
        // Adding myOwnMap method to Array prototype
        interface Array<T> {
          myOwnMap<U>(callback: (element: T, index: number) => U): U[];
          myOwnReduce(callback: (accumulator: T, currentValue: T, index?: number, array?: T[]) => T, initialValue?: T): T;
        }
        
        Array.prototype.myOwnMap = function <T, U>(callback: (element: T, index: number) => U): U[] {
          const newArr: U[] = [];
        
          this.forEach((element: T, index: number) => {
            const result = callback(element, index);
            newArr.push(result);
          });
        
          return newArr;
        };
        
        const arr = [1, 3, 2, 4, 9, 5, 8, 6];
        
        const arr2 = arr.myOwnMap((element, index) => {
          return element * 5;
        });
        
        console.log("Mapped Array:", arr2);
        
        ```
        
        1. Reduce
        
        ```tsx
        // Extending the Array interface to include myOwnReduce
        interface Array<T> {
          myOwnReduce<U>(
            callback: (accumulator: U, currentValue: T, index: number, array: T[]) => U,
            initialValue?: U
          ): U;
        }
        
        // Adding myOwnReduce method to Array prototype
        Array.prototype.myOwnReduce = function <T, U>(
          this: T[],
          callback: (accumulator: U, currentValue: T, index: number, array: T[]) => U,
          initialValue?: U
        ): U {
          if (this.length === 0 && initialValue === undefined) {
            throw new TypeError("Reduce of empty array with no initial value");
          }
        
          let accumulator: U = initialValue !== undefined ? initialValue : (this[0] as unknown as U);
          const startIndex = initialValue !== undefined ? 0 : 1;
        
          for (let i = startIndex; i < this.length; i++) {
            accumulator = callback(accumulator, this[i], i, this);
          }
        
          return accumulator;
        };
        
        // Example usage
        const arr = [1, 2, 3, 4];
        const sum = arr.myOwnReduce<number>((accumulator, currentValue) => {
          return accumulator + currentValue;
        }, 0);
        
        console.log("Reduced Sum:", sum); // Output: Reduced Sum: 10
        
        ```
        
    - Regex
        
        ```tsx
        const string: string = "all your string base belong to us";
        const regex: RegExp = /base/;
        const isExisting: boolean = regex.test(string);
        console.log(isExisting);
        
        ```
        
    - Sliding Window
        1. Longest Sub String No Repeating Char
        
        ```jsx
        let s: string = "abcabcbb";
        console.log(lengthOfLongestSubString(s));
        
        function lengthOfLongestSubString(s: string): number {
          let n = s.length;
          let left = 0;
          let right = 0;
          let charSet = new Set<string>(); // TypeScript: explicitly specifying the type of elements in the set
          let maxLen = 0;
        
          while (right < n) {
            if (!charSet.has(s[right])) {
              charSet.add(s[right]);
              maxLen = Math.max(maxLen, right - left + 1);
              right++;
            } else {
              charSet.delete(s[left]);
              left++;
            }
          }
        
          return maxLen;
        }
        
        ```
        
        1. Max Sub Array Size K
        
        ```jsx
        function slidingWindowFixed(arr: number[], k: number): number | null {
          const n = arr.length;
          if (n < k) return null;
        
          let left = 0;
          let right = 0;
          let maxSum = 0;
          let currSum = 0;
          const windowSet = new Set<number>();
        
          while (right < n) {
            if (!windowSet.has(arr[right]) && windowSet.size < k) {
              windowSet.add(arr[right]);
              currSum += arr[right];
              right++;
            } else {
              windowSet.delete(arr[left]);
              currSum -= arr[left];
              left++;
            }
        
            if (windowSet.size === k) {
              maxSum = Math.max(maxSum, currSum);
              windowSet.delete(arr[left]);
              currSum -= arr[left];
              left++;
            }
          }
          return maxSum;
        }
        function ifNotDistinct(arr: number[], k: number): number | null {
          const n = arr.length;
          if (n < k) return null;
        
          let left = 0;
          let right = 0;
          let maxSum = 0;
          let currentSum = 0;
        
          while (right < n) {
            currentSum += arr[right];
        
            if (right - left + 1 === k) {
              maxSum = Math.max(maxSum, currentSum);
              currentSum -= arr[left];
              left++;
            }
        
            right++;
          }
        
          return maxSum;
        }
        let arr = [1, 5, 4, 2, 9, 9, 9];
        let k = 3;
        
        console.log(slidingWindowFixed(arr, k)); // Output: Max sum of distinct elements within a sliding window
        console.log(ifNotDistinct(arr, k));     // Output: Max sum of a fixed-size sliding window
        
        ```
        
        1. First Negative Number of Size K
        
        ```jsx
        let arr: number[] = [-8, 2, 3, -6, 10];
        let k: number = 2;
        console.log(slidingWindowFixed(arr, k));
        
        function slidingWindowFixed(arr: number[], k: number): number[] {
            const n = arr.length;
            let left = 0;
            let right = 0;
            let window: number[] = [];
            let result: number[] = [];
        
            while (right < n) {
                // Add negative numbers to the window
                if (arr[right] < 0) {
                    window.push(arr[right]);
                }
        
                console.log(window); // Debugging: logging window contents
        
                // When the window size reaches 'k', process the window
                if (right - left + 1 === k) {
                    // If there are no negative numbers in the window, push 0 to the result
                    if (window.length === 0) {
                        result.push(0);
                    } else {
                        // Otherwise, push the first negative number in the window (i.e., window[0])
                        result.push(window[0]);
        
                        // If the element at the left of the window is the same as the first negative number,
                        // remove it from the window
                        if (arr[left] === window[0]) {
                            window.shift();
                        }
                    }
                    // Slide the window to the right
                    left++;
                }
                right++;
            }
        
            return result;
        }
        
        ```
        
        1. Max Of All Sub Array
        
        ```jsx
        function slidingWindowFixed(arr: number[], k: number): number[] {
          const n = arr.length;
          let left = 0;
          let right = 0;
          const ans: number[] = [];
          let maxNum = Number.MIN_VALUE;
        
          while (right < n) {
            // Update the maximum number in the current window
            if (arr[right] > maxNum) {
              maxNum = arr[right];
            }
        
            // Check if the window size equals k
            if (right - left + 1 === k) {
              ans.push(maxNum); // Add the maximum value to the result array
        
              // If the leftmost element was the maximum, recalculate for the new window
              if (arr[left] === maxNum) {
                maxNum = Number.MIN_VALUE;
                for (let i = left + 1; i <= right; i++) {
                  maxNum = Math.max(maxNum, arr[i]);
                }
              }
        
              left++; // Slide the window
            }
        
            right++; // Expand the window
          }
        
          return ans;
        }
        
        // Example Usage
        const arr: number[] = [1, 2, 3, 1, 4, 5, 2, 3, 6];
        const k: number = 3;
        console.log(slidingWindowFixed(arr, k)); // Output: [3, 3, 4, 5, 5, 5, 6]
        
        ```
        
        1. Max Consecutive longest Ones
        
        ```jsx
        function longestOnes(arr: number[], k: number): number {
          const n: number = arr.length;
          let left: number = 0;
          let right: number = 0;
          let window: number = 0; // Count of `1`s in the current window
          let ans: number = 0;
        
          while (right < n) {
            // Expand the window to include the current element
            window += arr[right];
        
            // If the window condition is violated, shrink from the left
            while (window + k < right - left + 1) {
              window -= arr[left];
              left++;
            }
        
            // Update the maximum size of the window
            ans = Math.max(ans, right - left + 1);
            right++;
          }
        
          return ans;
        }
        
        // Example Usage
        const arr: number[] = [
          0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1,
        ];
        const k: number = 3;
        console.log(longestOnes(arr, k)); // Output: 10
        
        ```
        
        1. Max Fruit
        
        ```jsx
        function findMaxFruits(arr: number[]): number {
          const n: number = arr.length;
          let left: number = 0;
          let right: number = 0;
          let ans: number = 0;
        
          const map: Map<number, number> = new Map();
        
          while (right < n) {
            // Add the current fruit to the map
            map.set(arr[right], (map.get(arr[right]) || 0) + 1);
        
            // If there are more than 2 distinct types of fruits, shrink the window
            while (map.size > 2) {
              const leftFruit = arr[left];
              const count = map.get(leftFruit) || 0;
        
              if (count === 1) {
                map.delete(leftFruit);
              } else {
                map.set(leftFruit, count - 1);
              }
              left++;
            }
        
            // Update the maximum length
            ans = Math.max(ans, right - left + 1);
        
            right++;
          }
        
          return ans;
        }
        
        // Example Usage
        const arr: number[] = [1, 2, 1, 1, 3, 4, 2, 2, 2, 2, 4];
        console.log(findMaxFruits(arr)); // Output: 5
        
        ```
        
        1. Character Replacement
        
        ```jsx
        function characterReplacement(s: string, k: number): number {
          const n: number = s.length;
          const charCount: number[] = new Array(26).fill(0);
          let maxCount: number = 0;
          let maxLen: number = 0;
          let left: number = 0;
        
          for (let right = 0; right < n; right++) {
            const charIndex: number = s.charCodeAt(right) - "A".charCodeAt(0);
            charCount[charIndex]++;
            maxCount = Math.max(maxCount, charCount[charIndex]);
        
            // Check if the current window is valid
            while (right - left + 1 - maxCount > k) {
              const leftIndex: number = s.charCodeAt(left) - "A".charCodeAt(0);
              charCount[leftIndex]--;
              left++;
            }
        
            // Update the maximum length
            maxLen = Math.max(maxLen, right - left + 1);
          }
        
          return maxLen;
        }
        
        // Example Usage
        const s: string = "AABABBA";
        const k: number = 1;
        console.log(characterReplacement(s, k)); // Output: 4
        
        ```
        
        1. Num SubArray with sum
        
        ```jsx
        function numSubarraysWithSum(nums: number[], goal: number): number {
          const prefixSumCount: Map<number, number> = new Map();
          let sum: number = 0;
          let count: number = 0;
        
          for (const num of nums) {
            // Add current number to the cumulative sum
            sum += num;
        
            // If the cumulative sum equals the goal, increment the count
            if (sum === goal) {
              count++;
            }
        
            // Check if there exists a prefix sum that satisfies the condition
            if (prefixSumCount.has(sum - goal)) {
              count += prefixSumCount.get(sum - goal)!;
            }
        
            // Update the prefixSumCount map
            prefixSumCount.set(sum, (prefixSumCount.get(sum) || 0) + 1);
          }
        
          return count;
        }
        
        // Example Usage
        const nums: number[] = [1, 0, 1, 0, 1];
        const goal: number = 2;
        console.log(numSubarraysWithSum(nums, goal)); // Output: 4
        
        ```
        
        1. Number of Sub Arrays
        
        ```jsx
        function numberOfSubarrays(nums: number[], k: number): number {
          const n: number = nums.length;
          let left: number = 0;
          let right: number = 0;
          let niceSubArray: number = 0;
          let result: number = 0;
          let countOdd: number = 0;
        
          while (right < n) {
            // Count odd numbers in the current window
            if (nums[right] % 2 !== 0) {
              countOdd++;
            }
        
            // When the current window has exactly k odd numbers
            if (countOdd === k) {
              niceSubArray = 0;
            }
        
            // Shrink the window from the left while maintaining k odd numbers
            while (countOdd >= k) {
              if (nums[left] % 2 !== 0) {
                countOdd--;
              }
              left++;
              niceSubArray++;
            }
        
            // Add the count of valid subarrays for the current window
            result += niceSubArray;
            right++;
          }
        
          return result;
        }
        
        // Example Usage
        const nums: number[] = [1, 1, 2, 1, 1];
        const k: number = 3;
        console.log(numberOfSubarrays(nums, k)); // Output: 2
        
        ```
        
        1. Number of substring
        
        ```jsx
        function numberOfSubstrings(s: string): number {
          const freq: Record<string, number> = { a: 0, b: 0, c: 0 };
          const n: number = s.length;
          let left: number = 0;
          let right: number = 0;
          let count: number = 0;
          let missing: number = 3; // Counts how many of 'a', 'b', 'c' are missing in the window
        
          while (right < n) {
            // Add the character at `right` to the frequency map
            if (++freq[s[right]] === 1) {
              missing--; // Decrease `missing` if a new character is added
            }
        
            // When all characters are present in the window
            while (missing === 0) {
              // All substrings starting from the current `left` to the end are valid
              count += n - right;
        
              // Remove the character at `left` from the window
              if (--freq[s[left]] === 0) {
                missing++; // Increase `missing` if a character is removed completely
              }
              left++; // Shrink the window
            }
        
            right++; // Expand the window
          }
        
          return count;
        }
        
        // Example Usage
        const s: string = "abcabc";
        console.log(numberOfSubstrings(s)); // Output: 10
        
        ```
        
    - Sorting
        
        Selection Sort
        
        ```tsx
        let arr: number[] = [13, 46, 24, 52, 20, 9];
        let n: number = arr.length;
        console.log("Selection Sort: ", selectionSort(arr, n));
        
        // TC: O(n^2) SC: O(1)
        function selectionSort(arr: number[], n: number): number[] {
          for (let i = 0; i < n - 1; i++) {
            let min: number = i;
            for (let j = i + 1; j < n; j++) {
              if (arr[j] < arr[min]) {
                min = j;
              }
            }
            let temp: number = arr[i];
            arr[i] = arr[min];
            arr[min] = temp;
          }
          return arr;
        }
        
        ```
        
        Quick Sort
        
        ```tsx
        const arr: number[] = [4, 6, 2, 5, 7, 9, 1, 3];
        console.log("Before Using Quicksort: ", arr);
        sortArray(arr);
        console.log("After Quicksort: ", arr);
        
        function partition(arr: number[], low: number, high: number): number {
          const pivot: number = arr[low];
          let i: number = low;
          let j: number = high;
        
          while (i < j) {
            while (arr[i] <= pivot && i <= high - 1) {
              i++;
            }
        
            while (arr[j] > pivot && j >= low + 1) {
              j--;
            }
        
            if (i < j) {
              const temp: number = arr[i];
              arr[i] = arr[j];
              arr[j] = temp;
            }
          }
        
          const temp: number = arr[low];
          arr[low] = arr[j];
          arr[j] = temp;
          return j;
        }
        
        function quickSort(arr: number[], low: number, high: number): void {
          if (low < high) {
            const pIndex: number = partition(arr, low, high);
            quickSort(arr, low, pIndex - 1);  // Recursively sort the left part
            quickSort(arr, pIndex + 1, high); // Recursively sort the right part
          }
        }
        
        function sortArray(arr: number[]): number[] {
          quickSort(arr, 0, arr.length - 1);
          return arr;
        }
        
        ```
        
        Merge Sort
        
        ```tsx
        let arr: number[] = [13, 46, 24, 52, 20, 9];
        let n: number = arr.length;
        mergeSort(arr, 0, n - 1);
        console.log("Merge sort: ", arr);
        
        // Time complexity: O(n log n)  // Space complexity: O(n)
        function mergeSort(arr: number[], low: number, high: number): void {
          if (low >= high) return;
          
          let mid: number = Math.floor((low + high) / 2);
          mergeSort(arr, low, mid);         // Sort the left half
          mergeSort(arr, mid + 1, high);    // Sort the right half
          merge(arr, low, mid, high);       // Merge both halves
        }
        
        function merge(arr: number[], low: number, mid: number, high: number): void {
          const temp: number[] = [];
          let left: number = low;
          let right: number = mid + 1;
        
          while (left <= mid && right <= high) {
            if (arr[left] <= arr[right]) {
              temp.push(arr[left]);
              left++;
            } else {
              temp.push(arr[right]);
              right++;
            }
          }
        
          // Add remaining elements from the left part
          while (left <= mid) {
            temp.push(arr[left]);
            left++;
          }
        
          // Add remaining elements from the right part
          while (right <= high) {
            temp.push(arr[right]);
            right++;
          }
        
          // Copy the sorted elements back into the original array
          for (let i = low; i <= high; i++) {
            arr[i] = temp[i - low];
          }
        }
        
        ```
        
        Insertion Sort
        
        ```tsx
        let arr:number[]=[23,23,43,34,5,6];
        let n:number=arr.length;
        insertionSort(arr,n);
        console.log("Insertion sort: ", arr);
        function insertionSort(arr:number[],n:number):number[]{
            for(let i=0;i<n;i++){
                let j=i;
                while(j>0 && arr[j-1]>arr[j]){
                    let temp=arr[j];
                    arr[j]=arr[j-1];
                    arr[j-1]=temp;
                    j--;
                }
            }
            return arr;
        }
        ```
        
        Buuble Sort
        
        ```tsx
        let arr: number[] = [12, 3, 2, 45, 34];
        let n: number = arr.length;
        bubbleSort(arr, n);
        console.log("Bubble Sort:", arr);
        
        function bubbleSort(arr: number[], n: number): void {
            for (let turn = 0; turn < n - 1; turn++) {
                for (let j = 0; j < n - turn - 1; j++) {  // Fixed range for `j`
                    if (arr[j] < arr[j + 1]) {  // Sort in descending order
                        let temp = arr[j];
                        arr[j] = arr[j + 1];
                        arr[j + 1] = temp;
                    }
                }
            }
        }
        
        ```
        
    - String
        1. Remove Outer most String
        
        ```tsx
        let s: string = "(()())(())";
        let ans: string = removeOuterParentheses(s);
        console.log("ans: ", ans);
        
        function removeOuterParentheses(s: string): string {
          let res: string = "";
          let bal: number = 0;
          
          for (let ch of s) {
            if (ch === "(") {
              if (bal > 0) {
                res += "(";
              }
              bal++;
            } else if (ch === ")") {
              bal--;
              if (bal > 0) {
                res += ")";
              }
            }
          }
        
          return res;
        }
        
        ```
        
        1. Reverse String
        
        ```tsx
        let str: string = "hello";
        let ans = reverse(str);
        console.log("ans: ", ans);
        
        function reverse(str: string): string {
          let strArr = str.split("");
          let low = 0;
          let high = strArr.length - 1; // Adjusted high index to length - 1
        
          while (low < high) {
            [strArr[low], strArr[high]] = [strArr[high], strArr[low]];
            low++;
            high--;
          }
        
          return strArr.join("");
        }
        
        ```
        
        1. Reverse Word
        
        ```tsx
        let str:string="this is and amazing program";
        console.log(reverseWords(str));
        function reverseWords(str:string){
            let reversedWord="";
            let reversedStr="";
             for (let i = 0; i < str.length; i++) {
            if (str[i] !== " ") {
              reversedWord = str[i] + reversedWord;
            } else {
              reversedStr += reversedWord + " ";
              reversedWord = "";
            }
          }
          // Handle the last word
          reversedStr += reversedWord;
          return reversedStr;
        }
        ```
        
        1. Duplicate Char String
        
        ```tsx
        let str: string = "test string";
        printDuplicate(str);
        console.log(str);
        
        function printDuplicate(str: string) {
            const charCount: { [key: string]: number } = {};  // Using an object to store character counts
            for (let i = 0; i < str.length; i++) {
                const char = str[i];  // Getting the character at index i
                if (charCount[char]) {
                    charCount[char]++;
                } else {
                    charCount[char] = 1;
                }
            }
        
            console.log("Character counts:", charCount);
            return charCount;
        }
        
        ```
        
        1. Odd Number In String
        
        ```tsx
        let num: string = "52";
        let largestOdd = largestOddNumber(num);
        console.log(largestOdd);  // This will now print the correct output
        
        function largestOddNumber(num: string): string {
            for (let i = num.length - 1; i >= 0; i--) {
                if (parseInt(num[i]) % 2 === 1) {
                    return num.substring(0, i + 1);  // Extracts the substring from the start to the first odd digit
                }
            }
            return "";  // If no odd digit found, return an empty string
        }
        
        ```
        
        1. Longest Common Prefix
        
        ```tsx
        let strs: string[] = ["flower", "flow", "flight"];
        let ans: string = longestCommonPrefix(strs);
        console.log("ans: ", ans);
        
        function longestCommonPrefix(strs: string[]): string {
          if (strs.length === 0) {
            return "";
          }
        
          const reference: string = strs[0];
        
          for (let i = 0; i < reference.length; i++) {
            const char: string = reference[i];
            for (let j = 1; j < strs.length; j++) {
              if (i >= strs[j].length || strs[j][i] !== char) {
                return reference.slice(0, i);
              }
            }
          }
          return reference;
        }
        
        ```
        
        1. Palindrome
        
        ```tsx
        let s: string = "abba";
        console.log(isPalindrome(s));  // Output will be true or false
        
        function isPalindrome(s: string): boolean {
            let left = 0, right = s.length - 1;
            while (left < right) {
                if (s[left] !== s[right]) {
                    return false;  // Return false if not a palindrome
                }
                left++;
                right--;
            }
            return true;  // Return true if palindrome
        }
        ```
        
        1. Isomorphic String
        
        ```tsx
        function isIsomorphic(s: string, t: string): boolean {
          if (s.length !== t.length) return false;
        
          const mpp = new Map<string, string>();
        
          for (let i = 0; i < s.length; i++) {
            const original = s[i];
            const replacement = t[i];
            if (!mpp.has(original)) {
              // Ensure no character in `t` is already mapped to another character in `s`
              if (![...mpp.values()].includes(replacement)) {
                mpp.set(original, replacement);
              } else {
                return false;
              }
            } else {
              const mappedChar = mpp.get(original);
              if (mappedChar !== replacement) {
                return false;
              }
            }
          }
          return true;
        }
        
        let s: string = "egg";
        let t: string = "add";
        console.log("ans: ", isIsomorphic(s, t));  // Output will be true or false
        
        ```
        
        1. Check if Valid Anagram
        
        ```tsx
        function isValidAnagramOptimal(s: string, t: string): boolean {
          // If lengths are different, they cannot be anagrams
          if (s.length !== t.length) {
            return false;
          }
        
          // Initialize a frequency array for characters (assuming uppercase English letters)
          let freq = new Array(26).fill(0);
        
          // Count the frequency of characters in s
          for (let i = 0; i < s.length; i++) {
            freq[s.charCodeAt(i) - "A".charCodeAt(0)]++;
          }
        
          // Subtract the frequency of characters in t
          for (let i = 0; i < t.length; i++) {
            freq[t.charCodeAt(i) - "A".charCodeAt(0)]--;
          }
        
          // Check if all frequencies are zero, meaning the strings are anagrams
          for (let i = 0; i < 26; i++) {
            if (freq[i] !== 0) {
              return false;
            }
          }
        
          return true;
        }
        
        // Example usage:
        let s: string = "INTEGER";
        let t: string = "TEGERNI";
        console.log(isValidAnagramOptimal(s, t));  // Output: true
        
        ```
        
        1. Sort Char Freq
        
        ```tsx
        function freqSort(s: string): string {
          let map = new Map<string, number>();  // Map to store character frequencies
          let str = "";
        
          // Populate the map with the frequency of each character
          for (let i = 0; i < s.length; i++) {
            if (!map.has(s[i])) {
              map.set(s[i], 1);
            } else {
              map.set(s[i], map.get(s[i])! + 1); // Using non-null assertion for the value
            }
          }
        
          // Sort the map entries by frequency in descending order
          const newMap = new Map([...map.entries()].sort((a, b) => b[1] - a[1]));
        
          // Build the final string based on the sorted frequencies
          for (let [i, j] of newMap) {
            str += i.repeat(j);
          }
        
          return str;
        }
        
        // Example usage:
        let s: string = "tree";
        console.log("ans: ", freqSort(s));  // Output: "eetr"
        
        ```
        
        1. Max Depth Parenthesis
        
        ```tsx
        function maxDepth(s: string): number {
          let maxDepth = 0;
          let currDept = 0;
        
          for (let i = 0; i < s.length; i++) {
            let ch = s[i];
            if (ch === "(") {
              currDept++;
              maxDepth = Math.max(maxDepth, currDept);
            } else if (ch === ")") {
              currDept--;
            }
          }
          return maxDepth;
        }
        
        // Example usage:
        let s: string = "(1+(2*3)+((8)/4))+1";
        console.log(maxDepth(s));  // Output: 3
        
        ```
        
        1. Roman To Integer
        
        ```tsx
        function romanToInt(s: string): number {
          let map = new Map<string, number>();
          map.set("I", 1);
          map.set("V", 5);
          map.set("X", 10);
          map.set("L", 50);
          map.set("C", 100);
          map.set("D", 500);
          map.set("M", 1000);
        
          let result = 0;
          for (let i = 0; i < s.length; i++) {
            let curr = map.get(s[i])!;
            let next = map.get(s[i + 1])!;
            if (curr < next) {
              result -= curr; // Subtract the current value if it's less than the next
            } else {
              result += curr; // Add the current value otherwise
            }
          }
          return result;
        }
        
        // Example usage:
        let s: string = "LVIII";
        console.log("ans: ", romanToInt(s));  // Output: 58
        
        ```
        
        1. Integer to Roman
        
        ```tsx
        function integerToRomanNaive(num: number): string {
          const map = new Map<number, string>();
          map.set(1, "I");
          map.set(5, "V");
          map.set(10, "X");
          map.set(50, "L");
          map.set(100, "C");
          map.set(500, "D");
          map.set(1000, "M");
        
          let base = 1;
          const result: string[] = [];
          while (num > 0) {
            const last = num % 10;
            if (last < 4) {
              for (let k = last; k > 0; k--) {
                result.unshift(map.get(base)!);
              }
            } else if (last == 4) {
              result.unshift(...[map.get(base)!, map.get(base * 5)!]);
            } else if (last == 5) {
              result.unshift(map.get(base * 5)!);
            } else if (last < 9) {
              for (let k = last; k > 5; k--) {
                result.unshift(map.get(base)!);
              }
              result.unshift(map.get(base * 5)!);
            } else {
              result.unshift(...[map.get(base)!, map.get(base * 10)!]);
            }
            base *= 10;
            num = (num - last) / 10;
          }
          return result.join("");
        }
        
        function integerToRomanOptimal(num: number): string {
          const map: [string, number][] = [
            ["M", 1000],
            ["CM", 900],
            ["D", 500],
            ["CD", 400],
            ["C", 100],
            ["XC", 90],
            ["L", 50],
            ["XL", 40],
            ["X", 10],
            ["IX", 9],
            ["V", 5],
            ["IV", 4],
            ["I", 1],
          ];
        
          let res = "";
        
          for (const [roman, val] of map) {
            while (num >= val) {
              res += roman;
              num -= val;
            }
          }
          return res;
        }
        
        // Example usage:
        let num: number = 58;
        console.log("ans: ", integerToRomanOptimal(num)); // Output: "LVIII"
        
        ```
        
        1. Count Distinct Character
        
        ```tsx
        let str: string = "aabab";
        let k: number = 2;
        console.log("ans: ", countSubstringsWithKDistinctCharsOptimal(str, k));
        
        function countSubstringsWithKDistinctCharsBruteForce(str: string, k: number): number {
          let n: number = str.length;
          let count: number = 0;
        
          for (let i = 0; i < n; i++) {
            for (let j = i; j < n; j++) {
              const substring: string = str.slice(i, j + 1);
              const distinctChars = new Set(substring);
              if (distinctChars.size === k) {
                count++;
              }
            }
          }
          return count;
        }
        
        function most_k_chars(s: string, k: number): number {
          if (!s) {
            return 0;
          }
        
          const char_count: { [key: string]: number } = {};
          let num: number = 0;
          let left: number = 0;
        
          for (let i = 0; i < s.length; i++) {
            char_count[s[i]] = (char_count[s[i]] || 0) + 1;
            while (Object.keys(char_count).length > k) {
              char_count[s[left]] -= 1;
              if (char_count[s[left]] === 0) {
                delete char_count[s[left]];
              }
              left += 1;
            }
            num += i - left + 1;
          }
          return num;
        }
        
        function countSubstringsWithKDistinctCharsOptimal(str: string, k: number): number {
          return most_k_chars(str, k) - most_k_chars(str, k - 1);
        }
        
        ```
        
        1. Longest Palindrome
        
        ```tsx
        let s: string = "babad";
        console.log(longestPalindromeOptimal(s));
        
        // TC: O(n^3), SC: O(1)
        function longestPalindrome(s: string): string {
          let n: number = s.length;
          let longest: string = "";
          for (let i = 0; i < n; i++) {
            for (let j = i + 1; j <= n; j++) {
              const substring: string = s.slice(i, j);
              if (isPalindrome(substring) && substring.length > longest.length) {
                longest = substring;
              }
            }
          }
          return longest;
        }
        
        function isPalindrome(str: string): boolean {
          const n: number = str.length;
          for (let i = 0; i < Math.floor(n / 2); i++) {
            if (str[i] !== str[n - 1 - i]) {
              return false;
            }
          }
          return true;
        }
        
        function expandOverCenter(str: string, left: number, right: number): string {
          let n: number = str.length;
          while (left >= 0 && right < n && str[left] === str[right]) {
            left--;
            right++;
          }
          return str.slice(left + 1, right);
        }
        
        function longestPalindromeOptimal(s: string): string {
          const n: number = s.length;
          let maxPalindrome: string = "";
          for (let i = 0; i < n; i++) {
            const palindrom1: string = expandOverCenter(s, i, i);
            const palindrom2: string = expandOverCenter(s, i, i + 1);
        
            if (palindrom1.length > maxPalindrome.length) {
              maxPalindrome = palindrom1;
            }
            if (palindrom2.length > maxPalindrome.length) {
              maxPalindrome = palindrom2;
            }
          }
          return maxPalindrome;
        }
        
        ```
        
        1. Sum of beauty
        
        ```tsx
        let s: string = "aabcb";
        console.log(sumOfBeautiesOptimal(s));
        
        function sumOfBeautiesNaive(s: string): number {
          let n: number = s.length;
          let beautySum: number = 0;
        
          for (let i = 0; i < n; i++) {
            for (let j = i + 1; j < n; j++) {
              const substring: string = s.slice(i, j);
              beautySum += calculateBeauty(substring);
            }
          }
          return beautySum;
        }
        
        function calculateBeauty(substring: string): number {
          const charCount: Map<string, number> = new Map();
          let maxCount: number = 0;
          let minCount: number = Number.MAX_VALUE;
        
          for (let i = 0; i < substring.length; i++) {
            const char: string = substring[i];
            charCount.set(char, (charCount.get(char) || 0) + 1);
            maxCount = Math.max(maxCount, charCount.get(char)!);
            minCount = Math.min(minCount, charCount.get(char)!);
          }
          return maxCount - minCount;
        }
        
        function sumOfBeautiesOptimal(s: string): number {
          let beautySum: number = 0;
          const limit: number = s.length;
        
          for (let i = 0; i < limit; i++) {
            const freq: number[] = new Array(26).fill(0);
            for (let j = i; j < limit; j++) {
              freq[s.charCodeAt(j) - "a".charCodeAt(0)]++;
              beautySum += calculateBeautyFreq(freq);
            }
          }
          return beautySum;
        }
        
        function calculateBeautyFreq(freq: number[]): number {
          let max: number = -Infinity;
          let min: number = Infinity;
        
          for (let i = 0; i < 26; i++) {
            if (freq[i] !== 0) {
              max = Math.max(max, freq[i]);
              min = Math.min(min, freq[i]);
            }
          }
          return max - min;
        }
        
        ```
        
        1. Atoi
        
        ```tsx
        function atoi(str: string): number {
          // Remove leading whitespace
          str = str.trim();
        
          // Check for empty string
          if (!str) return 0;
        
          // Initialize variables
          let sign: number = 1;
          let result: number = 0;
          let index: number = 0;
        
          // Determine sign
          if (str[index] === '-') {
            sign = -1;
            index++;
          } else if (str[index] === '+') {
            index++;
          }
        
          // Convert characters to integer until a non-digit is encountered
          while (index < str.length && str[index] >= '0' && str[index] <= '9') {
            const digit: number = str.charCodeAt(index) - '0'.charCodeAt(0);
            result = result * 10 + digit;
            index++;
            
            // Handle overflow for 32-bit signed integer range
            if (result * sign < -2147483648) return -2147483648;
            if (result * sign > 2147483647) return 2147483647;
          }
        
          return result * sign;
        }
        console.log(atoi("42"));           
        ```
        
    - Stack
        1. Stack Using Array
        
        ```tsx
        const myStack: string[] = [];
        myStack.push("a");
        myStack.push("b");
        myStack.push("c");
        console.log(myStack); // Output: ['a', 'b', 'c']
        
        myStack.pop();
        myStack.push("e");
        myStack.push("f");
        console.log(myStack); // Output: ['a', 'b', 'e', 'f']
        ```
        
        1. Stack Using LinkedList
        
        ```tsx
        class StackNode {
          value: string;
          next: StackNode | null;
        
          constructor(value: string) {
            this.value = value;
            this.next = null;
          }
        }
        
        class Stack {
          top: StackNode | null;
          size: number;
        
          constructor() {
            this.top = null;
            this.size = 0;
          }
        
          push(val: string): void {
            const newNode = new StackNode(val);
            if (this.size === 0) {
              this.top = newNode;
            } else {
              newNode.next = this.top;
              this.top = newNode;
            }
            this.size++;
          }
        
          getTop(): string | null {
            if (this.size === 0) return null;
            return this.top?.value || null;
          }
        
          pop(): string | null {
            if (this.size === 0) return null;
            const node = this.top;
            if (node) {
              this.top = node.next;
              this.size--;
              return node.value;
            }
            return null;
          }
        }
        
        // Testing the Stack
        const stack = new Stack();
        stack.push("a");
        stack.push("b");
        stack.push("c");
        stack.push("d");
        
        console.log(stack.pop()); // Output: "d"
        console.log(stack.pop()); // Output: "c"
        console.log(stack.pop()); // Output: "b"
        console.log(stack.pop()); // Output: "a"
        console.log(stack.getTop()); // Output: null
        ```
        
        1. Queue Using ArrayList
        
        ```tsx
        const queue:string[]=[];
        queue.push("a");
        queue.push("b");
        queue.push("c");
        queue.push("d");
        console.log(queue);
        queue.shift();
        console.log(queue);
        ```
        
        1. Queue using LinkedList
        
        ```tsx
        class QueueNode {
          value: string;
          next: QueueNode | null;
        
          constructor(value: string) {
            this.value = value;
            this.next = null;
          }
        }
        
        class Queue {
          front: QueueNode | null;
          back: QueueNode | null;
          size: number;
        
          constructor() {
            this.front = null;
            this.back = null;
            this.size = 0;
          }
        
          enque(value: string): void {
            const newNode = new QueueNode(value);
            if (this.size === 0) {
              this.front = newNode;
              this.back = newNode;
            } else {
              this.back!.next = newNode; // Use non-null assertion
              this.back = newNode;
            }
            this.size++;
          }
        
          deque(): string | null {
            if (this.size === 0) {
              return null;
            }
        
            const node = this.front; // Store the front node to return its value
            if (this.front !== null) { // Check if front is not null
              this.front = this.front.next;
            }
            if (this.size === 1) {
              this.back = null;
            }
            this.size--;
        
            return node!.value; // Use non-null assertion since node is not null here
          }
        }
        
        const queue: Queue = new Queue();
        queue.enque("a");
        queue.enque("b");
        queue.enque("c");
        queue.enque("d");
        console.log(queue);
        console.log(queue.front?.value);
        console.log(queue.back?.value);
        console.log(queue.deque());
        queue.deque();
        queue.deque();
        queue.deque();
        console.log(queue);
        
        ```
        
        1. Valid Paranthesis
        
        ```tsx
        let str: string = "(])";
        console.log(checkValidParenthesis(str));
        
        function checkValidParenthesis(str: string): boolean {
          let n: number = str.length;
          let stack: string[] = [];
          
          for (let i: number = 0; i < n; i++) {
            if (str[i] === "(" || str[i] === "[" || str[i] === "{") {
              stack.push(str[i]);
            } else {
              if (stack.length === 0) return false;
              let ch: string = stack[stack.length - 1];
              if (
                (str[i] === ")" && ch === "(") ||
                (str[i] === "}" && ch === "{") ||
                (str[i] === "]" && ch === "[")
              ) {
                stack.pop();
              } else {
                return false;
              }
            }
          }
          return stack.length === 0;
        }
        ```
        
        1. Stack Using Queue
        
        ```tsx
        class Stack {
          private queue1: number[];
          private queue2: number[];
        
          constructor() {
            this.queue1 = [];
            this.queue2 = [];
          }
        
          // Adds a new element to the stack
          push(value: number): void {
            this.queue1.push(value);
          }
        
          // Removes the top element from the stack and returns it
          pop(): number | null {
            if (this.queue1.length === 0) return null;
        
            // Transfer elements from queue1 to queue2, leaving only the last one
            while (this.queue1.length > 1) {
              this.queue2.push(this.queue1.shift() as number);
            }
        
            // The last element in queue1 is the top element of the stack
            const poppedItem = this.queue1.shift() as number;
        
            // Swap queues to keep queue1 as the main queue
            [this.queue1, this.queue2] = [this.queue2, this.queue1];
            return poppedItem;
          }
        
          // Returns the top element of the stack without removing it
          top(): number | null {
            if (this.queue1.length === 0) return null;
        
            // Transfer elements from queue1 to queue2, leaving only the last one
            while (this.queue1.length > 1) {
              this.queue2.push(this.queue1.shift() as number);
            }
        
            // Get the top item and add it back to queue2
            const topItem = this.queue1.shift() as number;
            this.queue2.push(topItem);
        
            // Swap queues to keep queue1 as the main queue
            [this.queue1, this.queue2] = [this.queue2, this.queue1];
            return topItem;
          }
        
          // Checks if the stack is empty
          isEmpty(): boolean {
            return this.queue1.length === 0;
          }
        
          // Returns the size of the stack
          size(): number {
            return this.queue1.length;
          }
        }
        
        // Testing the Stack implementation
        const stack: Stack = new Stack();
        stack.push(1);
        stack.push(2);
        stack.push(3);
        
        console.log(stack.pop()); // Output: 3
        console.log(stack.top()); // Output: 2
        console.log(stack.pop()); // Output: 2
        console.log(stack.isEmpty()); // Output: false
        console.log(stack.size()); // Output: 1
        
        ```
        
        1. Queue Using Stack
        
        ```jsx
        class Queue{
            private stack1:number[];
            private stack2: number[];
            constructor(){
                this.stack1=[];
                this.stack2=[];
            }
            enqueue(value:number):void{
                this.stack1.push(value);
            }
            dequeue():number|null{
                if(this.stack2.length===0){
                    if(this.stack1.length===0){
                        return null;
                    }
                    while(this.stack1.length>0){
                        this.stack2.push(this.stack1.pop() as number);
                    }
                }
                return this.stack2.pop() || null;
            }
            front():number | null{
                if(this.stack2.length===0){
                    if(this.stack1.length===0){
                        return null;
                    }
                    while(this.stack1.length>0){
                        this.stack2.push(this.stack1.pop()as number);
                    }
                }
                return this.stack2[this.stack2.length-1]||null;
            }
        isEmpty():boolean{
            return this.stack1.length===0&&this.stack2.length===0;
        }
        size():number{
            return this.stack1.length+this.stack2.length;
        }
        }
        // Example usage:
        const queue = new Queue();
        queue.enqueue(1);
        queue.enqueue(2);
        queue.enqueue(3);
        
        console.log(queue.dequeue()); // 1
        console.log(queue.front()); // 2
        console.log(queue.dequeue()); // 2
        console.log(queue.isEmpty()); // false
        console.log(queue.size()); // 1
        ```
        
        1. Min Stack
        
        ```jsx
        class MinStack{
            private stack:number[];
            private minStack:number[];
            constructor(){
                this.stack=[];
                this.minStack=[];
            }
        push(val:number):void{
            this.stack.push(val);
            if(this.minStack.length===0 || val<=this.minStack[this.minStack.length-1]){
                this.minStack.push(val);
            }
        }
        pop():void{
            if(this.stack.length===0)return;
            if(this.stack[this.stack.length-1]===this.minStack[this.minStack.length-1]){
                this.minStack.pop();
            }
            this.stack.pop();
        }
        top():number|null{
            if(this.stack.length===0) return null;
            return this.stack[this.stack.length-1];
        }
        getMin():number|null{
            if(this.minStack.length===0) return null;
            return this.minStack[this.minStack.length-1];
        }
        }
        const minStack = new MinStack();
        minStack.push(-2);
        minStack.push(0);
        minStack.push(-3);
        console.log(minStack.getMin()); // Output: -3
        minStack.pop();
        console.log(minStack.top());    // Output: 0
        console.log(minStack.getMin()); // Output: -2
        ```
        
        1. Next Greater Element
        
        ```jsx
        const nums1: number[] = [4, 1, 2];
        const nums2: number[] = [1, 3, 4, 2];
        const result = nextGreaterElement(nums1, nums2);
        console.log(result);
        
        function nextGreaterElement(nums1: number[], nums2: number[]): number[] {
            const nextGreater = new Map<number, number>();
            const stack: number[] = [];
        
            for (const num of nums2) {
                while (stack.length > 0 && stack[stack.length - 1] < num) {
                    nextGreater.set(stack.pop() as number, num);
                }
                stack.push(num);
            }
        
            const result: number[] = [];
            for (const num1 of nums1) {
                result.push(nextGreater.has(num1) ? nextGreater.get(num1)! : -1);
            }
            return result;
        }
        
        ```
        
        1. Next Smallest Element
        
        ```jsx
        let arr:number[]=[4,5,2,10,8];
        console.log(nextSmallerElement(arr));
        function nextSmallerElement(arr:number[]){
            let n:number=arr.length;
            let stack:number[]=[];
            let ans:number[]=[];
            ans[0]=-1;
        for(let i=0;i<n;i++){
            while(stack.length>0 && stack[stack.length-1]>=arr[i]){
                stack.pop() as number;
            }
            if(stack.length===0){
                ans[i]=-1;
            }
            else{
                ans[i]=stack[stack.length-1];
            }
            stack.push(arr[i]);
        }
        return ans;
        }
        ```
        
        1. Trapping Rain Water
        
        ```jsx
        let arr: number[] = [0, 1, 2, 1, 4, 3, 3, 2];
        let n: number = arr.length;
        console.log(trapWater(arr, n));
        
        function trapWater(arr: number[], n: number): number {
            let left = 0, right = n - 1;
            let waterTrapped = 0;
            let maxLeft = 0, maxRight = 0;
        
            while (left <= right) {
                if (arr[left] <= arr[right]) {
                    if (arr[left] >= maxLeft) {
                        maxLeft = arr[left];
                    } else {
                        waterTrapped += maxLeft - arr[left];
                    }
                    left++;
                } else {
                    if (arr[right] >= maxRight) {
                        maxRight = arr[right];
                    } else {
                        waterTrapped += maxRight - arr[right];
                    }
                    right--;
                }
            }
            return waterTrapped;
        }
        
        ```
        
        1. Sum of Sub Array Min
        
        ```jsx
        function subArrayMinOptimal(arr: number[], n: number): number {
          const mod = 10 ** 9 + 7;
          const left: number[] = new Array(n);
          const right: number[] = new Array(n);
          const stack: number[] = [];
        
          // Calculate the left boundary for each element
          for (let i = 0; i < n; i++) {
            while (stack.length > 0 && arr[stack[stack.length - 1]] > arr[i]) {
              stack.pop();
            }
            left[i] = stack.length === 0 ? -1 : stack[stack.length - 1];
            stack.push(i);
          }
        
          stack.length = 0; // Reset stack for right boundary calculation
        
          // Calculate the right boundary for each element
          for (let i = n - 1; i >= 0; i--) {
            while (stack.length !== 0 && arr[stack[stack.length - 1]] >= arr[i]) {
              stack.pop();
            }
            right[i] = stack.length === 0 ? n : stack[stack.length - 1];
            stack.push(i);
          }
        
          let sum = 0;
        
          // Calculate the sum of subarray minimums
          for (let i = 0; i < n; i++) {
            sum = (sum + (i - left[i]) * (right[i] - i) * arr[i]) % mod;
          }
        
          return sum;
        }
        
        // Example usage
        const arr: number[] = [3, 1, 2, 4];
        const n: number = arr.length;
        console.log(subArrayMinOptimal(arr, n)); // Output: 17
        
        ```
        
        1. Histogram
        
        ```jsx
        let arr:number[] = [2, 1, 5, 6, 2, 3, 1];
        let n:number = arr.length;
        console.log(largestArea(arr, n));
        
        function largestArea(arr:number[],n:number){
            let stack:number[]=[];
            let leftSmall:number[]=new Array(n);
            let rightSmall:number[]=new Array(n);
            for(let i=0;i<n;i++){
          while (stack.length != 0 && arr[stack[stack.length - 1]] >= arr[i]) {
              stack.pop();
            }
            if (stack.length == 0) {
              leftSmall[i] = 0;
            } else {
              leftSmall[i] = stack[stack.length - 1] + 1;
            }
            stack.push(i);
          }
        
          while (stack.length !== 0) {
            stack.pop();
          }
        
          for (let i = n - 1; i >= 0; i--) {
            while (stack.length !== 0 && arr[stack[stack.length - 1]] >= arr[i]) {
              stack.pop();
            }
            if (stack.length == 0) {
              rightSmall[i] = n - 1;
            } else {
              rightSmall[i] = stack[stack.length - 1] - 1;
            }
            stack.push(i);
          }
        
          let maxA = 0;
          for (let i = 0; i < n; i++) {
            maxA = Math.max(maxA, arr[i] * (rightSmall[i] - leftSmall[i] + 1));
          }
          return maxA;
        }
        ```
        
        1. Asteroid Collision
        
        ```jsx
        function asteroidCollision(asteroids: number[]): number[] {
          const stack: number[] = [];
        
          for (let i = 0; i < asteroids.length; i++) {
            let add = true;
            while (stack.length !== 0 && asteroids[i] < 0 && stack[stack.length - 1] > 0) {
              if (Math.abs(asteroids[i]) > Math.abs(stack[stack.length - 1])) {
                stack.pop();
              } else if (Math.abs(asteroids[i]) === Math.abs(stack[stack.length - 1])) {
                stack.pop();
                add = false;
                break;
              } else {
                add = false;
                break;
              }
            }
            if (add) stack.push(asteroids[i]);
          }
          return stack;
        }
        
        // Example usage
        let asteroids: number[] = [5, 10, -5];
        console.log(asteroidCollision(asteroids)); // Output: [5, 10]
        
        ```
        
        1. Sum of Sub Arrays Ranges
        
        ```jsx
        function subArrayRanges(arr: number[], n: number): number {
          let total = 0;
        
          for (let i = 0; i < n; i++) {
            let min = arr[i];
            let max = arr[i];
            for (let j = i; j < n; j++) {
              min = Math.min(min, arr[j]);
              max = Math.max(max, arr[j]);
              total += max - min;
            }
          }
          return total;
        }
        
        function subArrayRangesOptimal(arr: number[], n: number): number {
          const mod = 10 ** 9 + 7;
          let result = 0;
          let stack: number[] = [];
        
          const left: number[] = new Array(n).fill(0);
          const maxLeft: number[] = new Array(n).fill(0);
          const right: number[] = new Array(n).fill(0);
          const maxRight: number[] = new Array(n).fill(0);
        
          // Calculate left distances for minimums
          for (let i = 0; i < n; i++) {
            while (stack.length > 0 && arr[i] < arr[stack[stack.length - 1]]) {
              stack.pop();
            }
            left[i] = stack.length === 0 ? i + 1 : i - stack[stack.length - 1];
            stack.push(i);
          }
        
          stack = [];
        
          // Calculate right distances for minimums
          for (let i = n - 1; i >= 0; i--) {
            while (stack.length > 0 && arr[i] <= arr[stack[stack.length - 1]]) {
              stack.pop();
            }
            right[i] = stack.length === 0 ? n - i : stack[stack.length - 1] - i;
            stack.push(i);
          }
        
          stack = [];
        
          // Calculate left distances for maximums
          for (let i = 0; i < n; i++) {
            while (stack.length > 0 && arr[i] > arr[stack[stack.length - 1]]) {
              stack.pop();
            }
            maxLeft[i] = stack.length === 0 ? i + 1 : i - stack[stack.length - 1];
            stack.push(i);
          }
        
          stack = [];
        
          // Calculate right distances for maximums
          for (let i = n - 1; i >= 0; i--) {
            while (stack.length > 0 && arr[i] >= arr[stack[stack.length - 1]]) {
              stack.pop();
            }
            maxRight[i] = stack.length === 0 ? n - i : stack[stack.length - 1] - i;
            stack.push(i);
          }
        
          // Calculate the result using min and max distances
          for (let i = 0; i < n; i++) {
            result = (result + (maxLeft[i] * maxRight[i] - left[i] * right[i]) * arr[i]) % mod;
          }
        
          return result;
        }
        
        // Example usage:
        let arr: number[] = [1, 2, 3];
        let n: number = arr.length;
        console.log(subArrayRanges(arr, n)); // Brute force approach
        console.log(subArrayRangesOptimal(arr, n)); // Optimal approach
        
        ```
        
        1. Remove K Digits
        
        ```jsx
        function removeKdigits(num: string, k: number): string {
          let stack: string[] = [];
        
          for (let digit of num) {
            while (k > 0 && stack.length > 0 && digit < stack[stack.length - 1]) {
              stack.pop();
              k--;
            }
            stack.push(digit);
          }
        
          // Remove any remaining digits from the end if k > 0
          stack.length = Math.max(stack.length - k, 0);
        
          // Join the stack to form the result string and remove leading zeros
          let result = stack.join("");
          result = result.replace(/^0+/, "");
        
          // Return "0" if the result is empty, otherwise the result
          return result === "" ? "0" : result;
        }
        
        // Example usage:
        let num = "1432219";
        let k = 3;
        console.log(removeKdigits(num, k)); // Output: "1219"
        
        ```
        
        1. LRU Cache
        
        ```jsx
        class LRUCache {
          private capacity: number;
          private cache: Map<number, DoublyLinkedListNode>;
          private order: DoublyLinkedList;
        
          constructor(capacity: number) {
            this.capacity = capacity;
            this.cache = new Map(); // Use a Map for key-value storage
            this.order = new DoublyLinkedList(); // Doubly linked list for maintaining order
          }
        
          get(key: number): number {
            if (this.cache.has(key)) {
              // Move the accessed item to the front
              const node = this.cache.get(key)!;
              this.order.moveToFront(node);
              return node.value;
            }
            return -1;
          }
        
          put(key: number, value: number): void {
            if (this.cache.has(key)) {
              // Update the existing key
              const node = this.cache.get(key)!;
              node.value = value;
              this.order.moveToFront(node);
            } else {
              // Add a new key
              if (this.cache.size === this.capacity) {
                // Remove the least recently used item
                const removedKey = this.order.removeFromEnd();
                if (removedKey !== null) this.cache.delete(removedKey);
              }
              const newNode = new DoublyLinkedListNode(key, value);
              this.cache.set(key, newNode);
              this.order.addToFront(newNode);
            }
          }
        }
        
        class DoublyLinkedListNode {
          key: number;
          value: number;
          prev: DoublyLinkedListNode | null;
          next: DoublyLinkedListNode | null;
        
          constructor(key: number, value: number) {
            this.key = key;
            this.value = value;
            this.prev = null;
            this.next = null;
          }
        }
        
        class DoublyLinkedList {
          private head: DoublyLinkedListNode;
          private tail: DoublyLinkedListNode;
        
          constructor() {
            this.head = new DoublyLinkedListNode(0, 0);
            this.tail = new DoublyLinkedListNode(0, 0);
            this.head.next = this.tail;
            this.tail.prev = this.head;
          }
        
          addToFront(node: DoublyLinkedListNode): void {
            const next = this.head.next!;
            node.prev = this.head;
            node.next = next;
            this.head.next = node;
            next.prev = node;
          }
        
          removeFromEnd(): number | null {
            if (this.tail.prev === this.head) return null;
            const lastNode = this.tail.prev!;
            this.removeNode(lastNode);
            return lastNode.key;
          }
        
          moveToFront(node: DoublyLinkedListNode): void {
            this.removeNode(node);
            this.addToFront(node);
          }
        
          private removeNode(node: DoublyLinkedListNode): void {
            const prevNode = node.prev!;
            const nextNode = node.next!;
            prevNode.next = nextNode;
            nextNode.prev = prevNode;
          }
        }
        
        // Example usage:
        const lRUCache = new LRUCache(2);
        lRUCache.put(1, 1);
        lRUCache.put(2, 2);
        console.log(lRUCache.get(1)); // Output: 1
        lRUCache.put(3, 3); // LRU key was 2, evicts key 2
        console.log(lRUCache.get(2)); // Output: -1
        lRUCache.put(4, 4); // LRU key was 1, evicts key 1
        console.log(lRUCache.get(1)); // Output: -1
        console.log(lRUCache.get(3)); // Output: 3
        console.log(lRUCache.get(4)); // Output: 4
        
        ```
        
        - Greedy
            1. Fractional Knapsack
            
            ```jsx
            type Item = {
              value: number;
              weight: number;
            };
            
            function fractionalKnapsack(W: number, arr: Item[], n: number): number {
              // Sort items by value-to-weight ratio in descending order
              arr.sort((a, b) => b.value / b.weight - a.value / a.weight);
            
              let maxValue = 0; // Maximum value obtained
              let fractions: { item: Item; fraction: number }[] = []; // To track fractions of items used
            
              for (let item of arr) {
                if (item.weight <= W) {
                  // If the entire item fits, take it completely
                  maxValue += item.value;
                  W -= item.weight;
                  fractions.push({ item, fraction: 1 });
                } else {
                  // If only a part of the item fits, take the fraction
                  const fraction = W / item.weight;
                  maxValue += item.value * fraction;
                  fractions.push({ item, fraction });
                  break; // Knapsack is full
                }
              }
            
              // Optionally, you can return `fractions` for more details
              return maxValue;
            }
            
            // Example Usage
            const items: Item[] = [
              { value: 60, weight: 10 },
              { value: 100, weight: 20 },
              { value: 120, weight: 30 },
            ];
            
            const capacity = 50;
            console.log(fractionalKnapsack(capacity, items, items.length)); // Output: 240
            
            ```
            
            1. Fractional Knapsack
    - Recursion
        1. Atoi
        
        ```jsx
        function iterativeAtoi(str: string): number {
          let sign = 1;
          let base = 0;
          let i = 0;
        
          // Skip leading whitespace
          while (
            i < str.length &&
            (str[i] === " " || str[i] === "\t" || str[i] === "\n" || str[i] === "\r")
          ) {
            i++;
          }
        
          // Check for sign
          if (str[i] === "-" || str[i] === "+") {
            sign = str[i] === "-" ? -1 : 1;
            i++;
          }
        
          // Convert digits to integer
          while (i < str.length && str[i] >= "0" && str[i] <= "9") {
            if (
              base > Number.MAX_SAFE_INTEGER / 10 ||
              (base === Number.MAX_SAFE_INTEGER / 10 && +str[i] > 7)
            ) {
              return sign === 1 ? Number.MAX_SAFE_INTEGER : Number.MIN_SAFE_INTEGER;
            }
            base = 10 * base + (+str[i]);
            i++;
          }
        
          return base * sign;
        }
        
        function recursiveAtoi(str: string): number {
          let sign = 1;
          let base = 0;
          let i = 0;
        
          // Skip leading whitespace
          while (
            i < str.length &&
            (str[i] === " " || str[i] === "\t" || str[i] === "\n" || str[i] === "\r")
          ) {
            i++;
          }
        
          // Check for sign
          if (str[i] === "-" || str[i] === "+") {
            sign = str[i] === "-" ? -1 : 1;
            i++;
          }
        
          return convert(sign, str, i, base);
        }
        
        function convert(sign: number, str: string, i: number, base: number): number {
          if (i < str.length && str[i] >= "0" && str[i] <= "9") {
            if (
              base > Number.MAX_SAFE_INTEGER / 10 ||
              (base === Number.MAX_SAFE_INTEGER / 10 && +str[i] > 7)
            ) {
              return sign === 1 ? Number.MAX_SAFE_INTEGER : Number.MIN_SAFE_INTEGER;
            }
            base = 10 * base + (+str[i]);
            return convert(sign, str, i + 1, base);
          }
        
          return base * sign;
        }
        
        // Example Usage
        console.log(iterativeAtoi("-123")); // Output: -123
        console.log(iterativeAtoi("+123")); // Output: 123
        console.log(iterativeAtoi("/123")); // Output: 0 (invalid input)
        
        console.log(recursiveAtoi("-123")); // Output: -123
        console.log(recursiveAtoi("+123")); // Output: 123
        console.log(recursiveAtoi("/123")); // Output: 0 (invalid input)
        
        ```
        
        1. Pow XN
        
        ```jsx
        function myPowIterative(x: number, n: number): number {
          if (n === 0) return 1;
        
          let result = 1;
          let base = x;
          let exponent = Math.abs(n);
        
          while (exponent > 0) {
            if (exponent % 2 === 1) {
              result *= base;
            }
            base *= base;
            exponent = Math.floor(exponent / 2);
          }
        
          return n < 0 ? 1 / result : result;
        }
        
        function myPowRecursive(x: number, n: number): number {
          if (n === 0) return 1;
        
          if (n < 0) {
            x = 1 / x;
            n = -n;
          }
        
          return n % 2 === 0
            ? myPowRecursive(x * x, Math.floor(n / 2))
            : x * myPowRecursive(x * x, Math.floor(n / 2));
        }
        
        // Example Usage
        console.log(myPowIterative(2, 3));   // Output: 8
        console.log(myPowIterative(2, -3));  // Output: 0.125
        
        console.log(myPowRecursive(2, 3));   // Output: 8
        console.log(myPowRecursive(2, -3));  // Output: 0.125
        
        ```
        
        1. Count Good Number
        
        ```jsx
        const MOD = 10 ** 9 + 7;
        
        function countGoodDigitStrings(N: number): number {
          let count = 0;
        
          function isGoodDigitString(s: string): boolean {
            for (let i = 0; i < s.length; i++) {
              if (i % 2 === 0 && +s[i] % 2 !== 0) return false; // Even positions must have even digits
              if (i % 2 === 1 && ![2, 3, 5, 7].includes(+s[i])) return false; // Odd positions must have prime digits
            }
            return true;
          }
        
          for (let num = 0; num < Math.pow(10, N); num++) {
            const paddedNum = num.toString().padStart(N, "0");
            if (isGoodDigitString(paddedNum)) {
              count++;
            }
          }
        
          return count % MOD;
        }
        
        function countGoodDigitStringsOptimal(N: number): number {
          let count = 1;
        
          if (N % 2 === 0) {
            count *= modPow(4, Math.floor(N / 2)); // 4 choices for even positions
            count *= modPow(5, Math.floor(N / 2)); // 5 choices for odd positions
            count %= MOD;
          } else {
            count *= modPow(4, Math.floor(N / 2)); // 4 choices for even positions
            count *= modPow(5, Math.floor((N + 1) / 2)); // 5 choices for odd positions
            count %= MOD;
          }
        
          return count;
        }
        
        function modPow(base: number, exponent: number): number {
          if (exponent === 0) return 1;
        
          let result = 1;
          base = base % MOD;
        
          while (exponent > 0) {
            if (exponent % 2 === 1) {
              result = (result * base) % MOD;
            }
            base = (base * base) % MOD;
            exponent = Math.floor(exponent / 2);
          }
        
          return result % MOD;
        }
        
        // Example Usage:
        console.log(countGoodDigitStrings(2)); // Output for the brute-force approach
        console.log(countGoodDigitStringsOptimal(50)); // Output for the optimized approach
        
        ```
        
        1. Binary String With consecutive 1s 
        
        ```jsx
        function generateAllStrings(k: number): void {
          if (k <= 0) {
            return;
          }
          
          const str: string[] = new Array(k);
          
          // Start with the first character as "0"
          str[0] = "0";
          generateAllStringsUtil(k, str, 1);
        
          // Start with the first character as "1"
          str[0] = "1";
          generateAllStringsUtil(k, str, 1);
        }
        
        function generateAllStringsUtil(k: number, str: string[], n: number): void {
          // Base case: If the string length reaches k, print it
          if (n === k) {
            console.log(str.join(""));
            return;
          }
        
          // If the previous character is "1", only "0" can follow
          if (str[n - 1] === "1") {
            str[n] = "0";
            generateAllStringsUtil(k, str, n + 1);
          }
        
          // If the previous character is "0", both "0" and "1" can follow
          if (str[n - 1] === "0") {
            str[n] = "0";
            generateAllStringsUtil(k, str, n + 1);
        
            str[n] = "1";
            generateAllStringsUtil(k, str, n + 1);
          }
        }
        
        // Example usage
        const k = 3;
        generateAllStrings(k);
        
        ```
        
        1. Generate Paranthesis
        
        ```jsx
        function generateParenthesis(n: number): string[] {
          const result: string[] = [];
          generateAllParenthesis(result, "(", 1, 0, n);
          return result;
        }
        
        function generateAllParenthesis(
          result: string[],
          str: string,
          open: number,
          close: number,
          n: number
        ): void {
          // Base case: If the number of open and close parentheses equals n, add to result
          if (open === n && close === n) {
            result.push(str);
            return;
          }
        
          // If the number of open parentheses is less than n, add an open parenthesis
          if (open < n) {
            generateAllParenthesis(result, str + "(", open + 1, close, n);
          }
        
          // If the number of close parentheses is less than open, add a close parenthesis
          if (close < open) {
            generateAllParenthesis(result, str + ")", open, close + 1, n);
          }
        }
        
        // Example usage
        const n = 3;
        console.log(generateParenthesis(n));
        
        ```
        
        1. Seq of strings
        
        ```jsx
        const str: string = "abc";
        const ans: string[] = [];
        getAllSubsequence(0, str, ans, "");
        console.log("ans: ", ans);
        
        function getAllSubsequence(
          i: number,
          str: string,
          ans: string[],
          current: string
        ): void {
          // Base case: If the current index reaches the string's length, add the current subsequence to the result
          if (i === str.length) {
            ans.push(current);
            return;
          }
        
          // Include the current character in the subsequence
          getAllSubsequence(i + 1, str, ans, current + str[i]);
        
          // Exclude the current character from the subsequence
          getAllSubsequence(i + 1, str, ans, current);
        }
        
        ```
        
        1. Sub Sequence with sum k
        
        ```jsx
        const arr: number[] = [2, 5, 8, 4, 6, 11];
        const targetSum: number = 13;
        const ans: number[][] = [];
        const v: number[] = [];
        
        subSeq(0, arr, targetSum, ans, 0, v);
        console.log(ans);
        
        function subSeq(
          i: number,
          arr: number[],
          targetSum: number,
          ans: number[][],
          currSum: number,
          v: number[]
        ): void {
          // Base case: When the current index reaches the end of the array
          if (i === arr.length) {
            // If the current sum equals the target sum, add the current sequence to the result
            if (currSum === targetSum) {
              ans.push([...v]);
            }
            return;
          }
        
          // Include the current element in the subsequence
          subSeq(i + 1, arr, targetSum, ans, currSum + arr[i], [...v, arr[i]]);
        
          // Exclude the current element from the subsequence
          subSeq(i + 1, arr, targetSum, ans, currSum, v);
        }
        
        ```
        
        1. Find Combination Sum
        
        ```jsx
        const arr: number[] = [2, 3, 5];
        const target: number = 8;
        const ans: number[][] = [];
        
        findCombinationSum(0, arr, target, ans, []);
        console.log(ans);
        
        function findCombinationSum(
          i: number,
          arr: number[],
          target: number,
          ans: number[][],
          ds: number[]
        ): void {
          // Base case: If we have reached the end of the array
          if (i === arr.length) {
            // If the target is exactly 0, add the current combination to the result
            if (target === 0) {
              ans.push([...ds]);
            }
            return;
          }
        
          // Include the current element in the combination if it is less than or equal to the target
          if (arr[i] <= target) {
            ds.push(arr[i]); // Add the current element to the combination
            findCombinationSum(i, arr, target - arr[i], ans, ds); // Recurse with the reduced target
            ds.pop(); // Backtrack to explore other combinations
          }
        
          // Exclude the current element and move to the next
          findCombinationSum(i + 1, arr, target, ans, ds);
        }
        
        ```
        
        1. Combination
        
        ```jsx
        function findCombinationSum(
          i: number,
          arr: number[],
          target: number,
          ans: number[][],
          ds: number[]
        ): void {
          if (target === 0) {
            ans.push([...ds]); // Add the current combination to the results
            return;
          }
        
          for (let j = i; j < arr.length; j++) {
            // Skip duplicate elements to avoid duplicate combinations
            if (j > i && arr[j] === arr[j - 1]) continue;
        
            // Break the loop if the current number exceeds the target
            if (arr[j] > target) break;
        
            // Include the current element in the combination
            ds.push(arr[j]);
        
            // Recursively call for the next index with a reduced target
            findCombinationSum(j + 1, arr, target - arr[j], ans, ds);
        
            // Backtrack by removing the current element from the combination
            ds.pop();
          }
        }
        
        // Input
        const candidates: number[] = [10, 1, 2, 7, 6, 1, 5].sort((a, b) => a - b); // Sort the array
        const target: number = 8;
        const ans: number[][] = [];
        
        // Function call
        findCombinationSum(0, candidates, target, ans, []);
        
        // Output the results
        console.log(ans);
        
        ```
        
        1. Sub set sum 
        
        ```jsx
        // Function to calculate all subset sums
        function subSetSum(arr: number[], n: number): number[] {
          const ans: number[] = [];
          subSetSumHelper(0, arr, n, ans, 0);
          return ans;
        }
        
        function subSetSumHelper(
          index: number,
          arr: number[],
          n: number,
          ans: number[],
          sum: number
        ): void {
          if (index === n) {
            ans.push(sum); // Add the current sum to the answer array
            return;
          }
        
          // Include the current element in the sum
          subSetSumHelper(index + 1, arr, n, ans, sum + arr[index]);
        
          // Exclude the current element from the sum
          subSetSumHelper(index + 1, arr, n, ans, sum);
        }
        
        // Function to generate all subsets
        function subSet(
          index: number,
          arr: number[],
          n: number,
          ans: number[][],
          ds: number[]
        ): void {
          if (index === n) {
            ans.push([...ds]); // Add a copy of the current subset to the answer array
            return;
          }
        
          // Include the current element in the subset
          subSet(index + 1, arr, n, ans, [...ds, arr[index]]);
        
          // Exclude the current element from the subset
          subSet(index + 1, arr, n, ans, ds);
        }
        
        // Input array
        const arr: number[] = [3, 1, 2];
        const n: number = arr.length;
        
        // Calculate subset sums
        console.log("Subset sums: ", subSetSum(arr, n));
        
        // Generate subsets
        const subsets: number[][] = [];
        subSet(0, arr, n, subsets, []);
        console.log("Subsets: ", subsets);
        
        ```
        
        1. Subset sum2
        
        ```jsx
        function subsetWithDup(arr: number[], n: number): number[][] {
          const ans: number[][] = [];
          
          // Sort the array to ensure duplicates are adjacent
          arr.sort((a, b) => a - b);
          
          subSetHelper(0, arr, n, ans, []);
          
          return ans;
        }
        
        function subSetHelper(
          index: number,
          arr: number[],
          n: number,
          ans: number[][],
          ds: number[]
        ): void {
          // Add a copy of the current subset to the result
          ans.push([...ds]);
        
          // Loop through the array starting from the current index
          for (let i = index; i < n; i++) {
            // Skip duplicate elements at the same recursion level
            if (i !== index && arr[i] === arr[i - 1]) continue;
        
            // Include the current element in the subset
            ds.push(arr[i]);
        
            // Recursively call for the next index
            subSetHelper(i + 1, arr, n, ans, ds);
        
            // Backtrack: remove the last element
            ds.pop();
          }
        }
        
        // Input array
        const arr: number[] = [1, 2, 2];
        const n: number = arr.length;
        
        // Generate subsets
        console.log("Unique subsets: ", subsetWithDup(arr, n));
        
        ```
        
        1. combination sum3
        
        ```jsx
        function combinationSum3(k: number, n: number): number[][] {
          const ans: number[][] = [];
          
          // Call the helper function for backtracking
          backtrack(1, k, n, ans, [], 0);
          
          return ans;
        }
        
        function backtrack(
          index: number,
          k: number,
          n: number,
          ans: number[][],
          ds: number[],
          sum: number
        ): void {
          // If the subset has exactly `k` numbers and their sum equals `n`, add to the result
          if (ds.length === k && sum === n) {
            ans.push([...ds]);
            return;
          }
        
          // Stop recursion if the subset length exceeds `k` or the sum exceeds `n`
          if (ds.length >= k || sum > n) {
            return;
          }
        
          // Iterate from the current index to 9
          for (let i = index; i <= 9; i++) {
            // Include the current number in the subset
            ds.push(i);
        
            // Recursively call for the next index
            backtrack(i + 1, k, n, ans, ds, sum + i);
        
            // Backtrack: remove the last element
            ds.pop();
          }
        }
        
        // Input parameters
        const k: number = 3;
        const n: number = 7;
        
        // Get all combinations
        console.log("Combinations: ", combinationSum3(k, n));
        
        ```
        
        1. Letter combination
        
        ```jsx
        const digit: string = "234";
        
        const map: { [key: string]: string } = {
          2: "abc",
          3: "def",
          4: "ghi",
          5: "jkl",
          6: "mno",
          7: "pqrs",
          8: "tuv",
          9: "wxyz",
        };
        
        console.log(letterCombination(digit));
        
        function letterCombination(digit: string): string[] {
          if (!digit) return [];
          
          const ans: string[] = [];
          
          // Call the helper function for backtracking
          backtrack(0, ans, digit, "");
          
          return ans;
        }
        
        function backtrack(
          i: number,
          ans: string[],
          digit: string,
          curr: string
        ): void {
          // If the current index reaches the length of digits, add the combination
          if (i === digit.length) {
            ans.push(curr);
            return;
          }
        
          // Iterate over the letters mapped to the current digit
          for (const letter of map[digit[i]]) {
            backtrack(i + 1, ans, digit, curr + letter);
          }
        }
        
        ```
        
        1. Palindrome Partitioning
        
        ```jsx
        const s: string = "aabb";
        console.log(palindromePartition(s));
        
        function palindromePartition(s: string): string[][] {
          const ans: string[][] = [];
          backtrack(0, s, ans, []);
          return ans;
        }
        
        function backtrack(
          index: number,
          s: string,
          ans: string[][],
          ds: string[]
        ): void {
          // Base case: if the current index reaches the end of the string
          if (index === s.length) {
            ans.push([...ds]);
            return;
          }
        
          // Explore all substrings starting from the current index
          for (let i = index; i < s.length; i++) {
            if (isPalindrome(s, index, i)) {
              ds.push(s.substring(index, i + 1)); // Add the palindrome substring
              backtrack(i + 1, s, ans, ds); // Recurse for the next part of the string
              ds.pop(); // Backtrack
            }
          }
        }
        
        function isPalindrome(s: string, start: number, end: number): boolean {
          while (start < end) {
            if (s[start++] !== s[end--]) {
              return false;
            }
          }
          return true;
        }
        
        ```
        
        1. Word Search
        
        ```jsx
        const board: string[][] = [
          ["A", "B", "C", "E"],
          ["S", "F", "C", "S"],
          ["A", "D", "E", "E"],
        ];
        const word: string = "SEE";
        
        console.log(wordSearch(board, word));
        
        function wordSearch(board: string[][], word: string): boolean {
          const n: number = board.length;
          const m: number = board[0].length;
        
          for (let i = 0; i < n; i++) {
            for (let j = 0; j < m; j++) {
              if (board[i][j] === word[0]) {
                if (searchNext(board, word, i, j, 0, n, m)) {
                  return true;
                }
              }
            }
          }
          return false;
        }
        
        function searchNext(
          board: string[][],
          word: string,
          row: number,
          col: number,
          index: number,
          n: number,
          m: number
        ): boolean {
          // Base case: if we've matched the entire word
          if (index === word.length) {
            return true;
          }
        
          // Boundary and validity checks
          if (
            row < 0 ||
            col < 0 ||
            row >= n ||
            col >= m ||
            board[row][col] !== word[index] ||
            board[row][col] === "!"
          ) {
            return false;
          }
        
          // Save the current character and mark it as visited
          const temp: string = board[row][col];
          board[row][col] = "!";
        
          // Explore in all 4 directions
          const top = searchNext(board, word, row - 1, col, index + 1, n, m);
          const bottom = searchNext(board, word, row + 1, col, index + 1, n, m);
          const left = searchNext(board, word, row, col - 1, index + 1, n, m);
          const right = searchNext(board, word, row, col + 1, index + 1, n, m);
        
          // Restore the current cell
          board[row][col] = temp;
        
          // Return true if any direction leads to a solution
          return top || bottom || left || right;
        }
        
        ```
        
        1. NQueens
        
        ```jsx
        const n: number = 4;
        console.log(solveNQueens(n));
        
        function solveNQueens(n: number): string[][] {
          const board: string[][] = Array.from({ length: n }, () => Array(n).fill("."));
          const ans: string[][] = [];
          backtrack(0, board, ans);
          return ans;
        }
        
        function backtrack(col: number, board: string[][], ans: string[][]): void {
          if (col === board.length) {
            ans.push(board.map((row) => row.join("")));
            return;
          }
        
          for (let row = 0; row < board.length; row++) {
            if (validate(board, row, col)) {
              board[row][col] = "Q";
              backtrack(col + 1, board, ans);
              board[row][col] = ".";
            }
          }
        }
        
        function validate(board: string[][], row: number, col: number): boolean {
          let tempRow = row;
          let tempCol = col;
        
          // Check upper diagonal (top-left)
          while (row >= 0 && col >= 0) {
            if (board[row][col] === "Q") return false;
            row--;
            col--;
          }
        
          row = tempRow;
          col = tempCol;
        
          // Check left side
          while (col >= 0) {
            if (board[row][col] === "Q") return false;
            col--;
          }
        
          row = tempRow;
          col = tempCol;
        
          // Check lower diagonal (bottom-left)
          while (col >= 0 && row < board.length) {
            if (board[row][col] === "Q") return false;
            col--;
            row++;
          }
        
          return true;
        }
        
        ```
        
        1. ratInMaze
        
        ```jsx
        const maze: number[][] = [
          [1, 0, 0, 0],
          [1, 1, 0, 1],
          [1, 1, 0, 0],
          [0, 1, 1, 1],
        ];
        const n: number = maze.length;
        const result = findPaths(n, maze);
        console.log(result);
        
        function findPaths(n: number, m: number[][]): string[] {
          const paths: string[] = [];
          const path: string[] = new Array(n).fill("");
          const directions: string[] = ["D", "L", "R", "U"];
          const visited: boolean[][] = Array.from({ length: n }, () => Array(n).fill(false));
        
          function isSafe(x: number, y: number, visited: boolean[][]): boolean {
            return x >= 0 && x < n && y >= 0 && y < n && m[x][y] === 1 && !visited[x][y];
          }
        
          function findPath(x: number, y: number, visited: boolean[][], index: number): void {
            if (x === n - 1 && y === n - 1) {
              paths.push(path.join(""));
              return;
            }
        
            visited[x][y] = true;
        
            for (const dir of directions) {
              const newX: number = x + (dir === "U" ? -1 : dir === "D" ? 1 : 0);
              const newY: number = y + (dir === "L" ? -1 : dir === "R" ? 1 : 0);
        
              if (isSafe(newX, newY, visited)) {
                path[index] = dir;
                findPath(newX, newY, visited, index + 1);
              }
            }
        
            visited[x][y] = false;
          }
        
          findPath(0, 0, visited, 0);
        
          return paths.sort();
        }
        
        ```
        
        1. word break
        
        ```jsx
        const s1: string = "leetcode";
        const wordDict1: string[] = ["leet", "code"];
        console.log(wordBreakRecursion(s1, wordDict1)); // Output: true
        
        const s2: string = "applepenapple";
        const wordDict2: string[] = ["apple", "pen"];
        console.log(wordBreakRecursion(s2, wordDict2)); // Output: true
        
        const s3: string = "catsandog";
        const wordDict3: string[] = ["cats", "dog", "sand", "and", "cat"];
        console.log(wordBreakRecursion(s3, wordDict3)); // Output: false
        
        function wordBreakRecursion(s: string, wordDict: string[]): boolean {
          function canSegment(s: string): boolean {
            if (s === "") {
              return true;
            }
        
            for (const word of wordDict) {
              if (s.startsWith(word) && canSegment(s.slice(word.length))) {
                return true;
              }
            }
            return false;
          }
        
          return canSegment(s);
        }
        
        ```
        
        1. nColoring
        
        ```jsx
        const N: number = 4;
        const M: number = 3;
        const graph: number[][] = [
          [0, 1, 1, 1],
          [1, 0, 1, 0],
          [1, 1, 0, 1],
          [1, 0, 1, 0],
        ];
        console.log(graphColoring(graph, M, N));
        
        function graphColoring(graph: number[][], M: number, N: number): boolean {
          const color: number[] = new Array(N).fill(-1);
        
          function isSafe(vertex: number, c: number): boolean {
            for (let i = 0; i < N; i++) {
              if (graph[vertex][i] === 1 && color[i] === c) {
                return false;
              }
            }
            return true;
          }
        
          function graphColorUtil(vertex: number): boolean {
            if (vertex === N) {
              return true;
            }
        
            for (let c = 1; c <= M; c++) {
              if (isSafe(vertex, c)) {
                color[vertex] = c;
                if (graphColorUtil(vertex + 1)) {
                  return true;
                }
                color[vertex] = -1; // Backtrack
              }
            }
            return false;
          }
        
          return graphColorUtil(0);
        }
        
        ```
        
        1. sudoku
        
        ```jsx
        const board = [
          ["5", "3", ".", ".", "7", ".", ".", ".", "."],
          ["6", ".", ".", "1", "9", "5", ".", ".", "."],
          [".", "9", "8", ".", ".", ".", ".", "6", "."],
          ["8", ".", ".", ".", "6", ".", ".", ".", "3"],
          ["4", ".", ".", "8", ".", "3", ".", ".", "1"],
          ["7", ".", ".", ".", "2", ".", ".", ".", "6"],
          [".", "6", ".", ".", ".", ".", "2", "8", "."],
          [".", ".", ".", "4", "1", "9", ".", ".", "5"],
          [".", ".", ".", ".", "8", ".", ".", "7", "9"],
        ];
        
        solveSudoku(board);
        console.log(board);
        
        function solveSudoku(board) {
          for (let i = 0; i < 9; i++) {
            for (let j = 0; j < 9; j++) {
              if (board[i][j] === ".") {
                for (let c = "1"; c <= "9"; c++) {
                  if (isValid(board, i, j, c)) {
                    board[i][j] = c.toString();
                    if (solveSudoku(board)) return true;
                    else board[i][j] = ".";
                  }
                }
                return false; // No valid number found for this cell
              }
            }
          }
          return true; // All cells are filled
        }
        
        function isValid(board, row, col, c) {
          c = c.toString(); // Convert c to a string for comparison
          for (let i = 0; i < 9; i++) {
            if (board[i][col] === c) {
              return false; // Check column
            }
            if (board[row][i] === c) {
              return false; // Check row
            }
            if (
              board[3 * Math.floor(row / 3) + Math.floor(i / 3)][
                3 * Math.floor(col / 3) + (i % 3)
              ] === c
            ) {
              return false; // Check 3x3 subgrid
            }
          }
          return true; // No conflicts found, it's valid
        }
        
        ```
        
    - Matrix
        1. Spiral Traversal
        
        ```jsx
        function printSpiral(mat: number[][]): number[] {
          let ans: number[] = [];
          let n: number = mat.length;
          let m: number = mat[0].length;
          let top: number = 0,
            left: number = 0,
            right: number = m - 1,
            bottom: number = n - 1;
        
          while (top <= bottom && left <= right) {
            // Traverse from left to right
            for (let i = left; i <= right; i++) {
              ans.push(mat[top][i]);
            }
            top++;
        
            // Traverse from top to bottom
            for (let i = top; i <= bottom; i++) {
              ans.push(mat[i][right]);
            }
            right--;
        
            // Traverse from right to left
            if (top <= bottom) {
              for (let i = right; i >= left; i--) {
                ans.push(mat[bottom][i]);
              }
              bottom--;
            }
        
            // Traverse from bottom to top
            if (left <= right) {
              for (let i = bottom; i >= top; i--) {
                ans.push(mat[i][left]);
              }
              left++;
            }
          }
        
          return ans;
        }
        
        // Example matrix
        let mat: number[][] = [
          [1, 2, 3, 4],
          [5, 6, 7, 8],
          [9, 10, 11, 12],
          [13, 14, 15, 16],
        ];
        
        // Call function and log result
        let ans: number[] = printSpiral(mat);
        for (let i = 0; i < ans.length; i++) {
          console.log(ans[i] + " ");
        }
        
        ```
        
        1. Search Element Matrix
        
        ```jsx
        const mat: number[][] = [
          [1, 2, 3, 4],
          [5, 6, 7, 8],
          [9, 10, 11, 12],
        ];
        
        let n: number = mat.length;
        let m: number = mat[0].length;
        let target: number = 8;
        
        let ans: number[] = searchMatrixOptimal(mat, n, m, target);
        console.log("ans: ", ans);
        
        // TC: O(n*m) SC: O(1)
        function searchMatrixNaive(mat: number[][], n: number, m: number, target: number): number[] {
          for (let i = 0; i < n; i++) {
            for (let j = 0; j < m; j++) {
              if (mat[i][j] === target) {
                return [i, j];
              }
            }
          }
          return [-1, -1];
        }
        
        // TC: O(N + logM) SC: O(1)
        function searchMatrixBetter(mat: number[][], n: number, m: number, target: number): number[] {
          for (let i = 0; i < n; i++) {
            if (mat[i][0] <= target && target <= mat[i][m - 1]) {
              return binarySearch(mat[i], target, i);
            }
          }
          return [-1, -1];
        }
        
        function binarySearch(arr: number[], target: number, i: number): number[] {
          let n: number = arr.length;
          let low: number = 0,
            high: number = n - 1;
          while (low <= high) {
            let mid: number = Math.floor((low + high) / 2);
            if (arr[mid] === target) {
              return [i, mid];
            } else if (target > arr[mid]) {
              low = mid + 1;
            } else {
              high = mid - 1;
            }
          }
          return [-1, -1];
        }
        
        // TC: O(log(n*m)) SC: O(1)
        function searchMatrixOptimal(arr: number[][], n: number, m: number, target: number): number[] {
          let low: number = 0;
          let high: number = n * m - 1;
        
          while (low <= high) {
            let mid: number = Math.floor((low + high) / 2);
            let row: number = Math.floor(mid / m);
            let col: number = mid % m;
            if (arr[row][col] === target) return [row, col];
            else if (arr[row][col] < target) low = mid + 1;
            else high = mid - 1;
          }
        
          return [-1, -1];
        }
        
        ```
        
        1. Median Row Matrix
        
        ```jsx
        let arr: number[][] = [
          [1, 3, 8],
          [2, 3, 4],
          [1, 2, 5],
        ];
        let row: number = arr.length;
        let col: number = arr[0].length;
        let ans: number = findMedianOptimal(arr, row, col);
        console.log("ans", ans);
        
        // Naive approach: O(row * col * log(row * col))
        function findMedianNaive(arr: number[][], row: number, col: number): number {
          let median: number[] = new Array(row * col);
        
          let index: number = 0;
          for (let i = 0; i < row; i++) {
            for (let j = 0; j < col; j++) {
              median[index] = arr[i][j];
              index++;
            }
          }
        
          median.sort((a, b) => a - b);
          return median[Math.floor((row * col) / 2)];
        }
        
        // Optimized approach: O(row * log(col))
        function findMedianOptimal(arr: number[][], row: number, col: number): number {
          let low: number = 1;
          let high: number = 1000000000;
          let n: number = row;
          let m: number = col;
          while (low <= high) {
            let mid: number = Math.floor((low + high) >> 1);
            let cnt: number = 0;
            for (let i = 0; i < n; i++) {
              cnt += countSmallerThanMid(arr[i], mid, col);
            }
            if (cnt <= (n * m) / 2) low = mid + 1;
            else high = mid - 1;
          }
          return low;
        }
        
        // Helper function to count elements smaller than or equal to `mid` in a sorted row
        function countSmallerThanMid(arr: number[], mid: number, n: number): number {
          let l: number = 0;
          let h: number = n - 1;
          while (l <= h) {
            let md: number = (l + h) >> 1;
            if (arr[md] <= mid) {
              l = md + 1;
            } else {
              h = md - 1;
            }
          }
          return l;
        }
        
        ```
        
        1. Row With Max 1s
        
        ```jsx
        const matrix: number[][] = [
          [1, 1, 1],
          [0, 0, 1],
          [0, 0, 0],
        ];
        const n: number = 3;
        const m: number = 3;
        
        console.log(
          "The row with maximum no. of 1's is: " + rowWithMax1Optimal(matrix, n, m)
        );
        
        // Naive approach: O(n * m) SC: O(1)
        function rowWithMax1Naive(mat: number[][], n: number, m: number): number {
          let maxCount: number = 0;
          let index: number = -1;
          for (let i = 0; i < n; i++) {
            let count: number = 0;
            for (let j = 0; j < m; j++) {
              if (mat[i][j] === 1) {
                count++;
              }
            }
            if (count > maxCount) {
              maxCount = count;
              index = i;
            }
          }
          return index;
        }
        
        // Helper function: Binary search to find the lower bound of 1 in a row
        function lowerBound(arr: number[], n: number, target: number): number {
          let low: number = 0;
          let high: number = n - 1;
          let ans: number = n;
          while (low <= high) {
            let mid: number = Math.floor((low + high) / 2);
            if (arr[mid] >= target) {
              ans = mid;
              high = mid - 1;
            } else {
              low = mid + 1;
            }
          }
          return ans;
        }
        
        // Optimized approach: O(n * log m) SC: O(1)
        function rowWithMax1Optimal(mat: number[][], n: number, m: number): number {
          let maxCount: number = 0;
          let index: number = -1;
          for (let i = 0; i < n; i++) {
            // Count the number of 1's in the row using binary search
            let count: number = m - lowerBound(mat[i], m, 1);
            if (count > maxCount) {
              maxCount = count;
              index = i;
            }
          }
          return index;
        }
        
        ```
        
        1. Sorted Matrix
        
        ```jsx
        let mat: number[][] = [
          [40, 94, 73, 98, 27],
          [58, 89, 87, 95, 9],
          [95, 28, 34, 74, 32],
          [19, 46, 78, 64, 80],
          [72, 62, 86, 16, 99],
        ];
        
        let n: number = mat.length;
        console.log("before: ", mat);
        sortMatrixOptimal(mat, n);
        console.log("after: ", mat);
        
        function sortMatrix(mat: number[][], n: number): void {
          let temp: number[] = new Array(n * n);
          let k: number = 0;
          for (let i = 0; i < n; i++) {
            for (let j = 0; j < n; j++) {
              temp[k++] = mat[i][j];
            }
          }
          temp.sort((a, b) => a - b);
          k = 0;
          for (let i = 0; i < n; i++) {
            for (let j = 0; j < n; j++) {
              mat[i][j] = temp[k++];
            }
          }
        }
        
        function sortMatrixOptimal(matrix: number[][], n: number): number[][] {
          const numRows: number = matrix.length;
          const numCols: number = matrix[0].length;
          let flatArray: number[] = [].concat(...matrix);
          let index: number = 0;
          flatArray.sort((a, b) => a - b);
          let res: number[][] = new Array(numRows);
        
          for (let i = 0; i < numRows; i++) {
            for (let j = 0; j < numCols; j++) {
              matrix[i][j] = flatArray[index];
              index++;
            }
          }
          return matrix;
        }
        
        function sortMatrixRowWiseColWise(mat: number[][], n: number): void {
          sortByRow(mat, n);
          transpose(mat, n);
          sortByRow(mat, n);
          transpose(mat, n);
        }
        
        function sortByRow(mat: number[][], n: number): void {
          for (let i = 0; i < n; i++) {
            mat[i].sort((a, b) => a - b);
          }
        }
        
        function transpose(mat: number[][], n: number): void {
          for (let i = 0; i < n; i++) {
            for (let j = i + 1; j < n; j++) {
              let temp: number = mat[i][j];
              mat[i][j] = mat[j][i];
              mat[j][i] = temp;
            }
          }
        }
        
        ```
        
        1. Find Pair Matrix
        
        ```jsx
        let mat: number[][] = [
          [1, 2, -1, -4, -20],
          [-8, -3, 4, 2, 1],
          [3, 8, 6, 1, 3],
          [-4, -1, 1, 7, -6],
          [0, -4, 10, -5, 1],
        ];
        let n: number = mat.length;
        let ans: number = findPairOptimal(mat, n);
        console.log("ans: ", ans);
        
        // TC: O(n^4) SC: O(1)
        function findPairNaive(mat: number[][], n: number): number {
          let maxVal: number = Number.MIN_VALUE;
          for (let i = 0; i < n - 1; i++) {
            for (let j = 0; j < n - 1; j++) {
              for (let k = i + 1; k < n; k++) {
                for (let l = j + 1; l < n; l++) {
                  if (maxVal < mat[k][l] - mat[i][j]) {
                    maxVal = mat[k][l] - mat[i][j];
                  }
                }
              }
            }
          }
          return maxVal;
        }
        
        function findPairOptimal(mat: number[][], n: number): number {
          let maxValue: number = Number.MIN_VALUE;
          let maxArr: number[][] = new Array(n);
        
          for (let i = 0; i < n; i++) {
            maxArr[i] = new Array(n);
          }
        
          maxArr[n - 1][n - 1] = mat[n - 1][n - 1];
          let maxv: number = mat[n - 1][n - 1];
        
          for (let j = n - 2; j >= 0; j--) {
            if (mat[n - 1][j] > maxv) maxv = mat[n - 1][j];
            maxArr[n - 1][j] = maxv;
          }
        
          maxv = mat[n - 1][n - 1];
          for (let i = n - 2; i >= 0; i--) {
            if (mat[i][n - 1] > maxv) maxv = mat[i][n - 1];
            maxArr[i][n - 1] = maxv;
          }
        
          for (let i = n - 2; i >= 0; i--) {
            for (let j = n - 2; j >= 0; j--) {
              // Update maxValue
              if (maxArr[i + 1][j + 1] - mat[i][j] > maxValue)
                maxValue = maxArr[i + 1][j + 1] - mat[i][j];
        
              // Set maxArr (i, j)
              maxArr[i][j] = Math.max(
                mat[i][j],
                Math.max(maxArr[i][j + 1], maxArr[i + 1][j])
              );
            }
          }
        
          return maxValue;
        }
        
        ```
        
        1. Rotate by 90
        
        ```jsx
        let arr: number[][] = [
          [1, 2, 3],
          [4, 5, 6],
          [7, 8, 9],
        ];
        let rotated: number[][] = rotateMatOptimal(arr);
        console.log("rotated matrix: ", rotated);
        
        // TC: O(n^2) SC: O(n^2)
        function rotateMatNaive(arr: number[][]): number[][] {
          let n: number = arr.length;
          let ans: number[][] = [[], [], []];
          for (let i = 0; i < n; i++) {
            for (let j = 0; j < n; j++) {
              ans[j][n - i - 1] = arr[i][j];
            }
          }
          return ans;
        }
        
        // TC: O(n^2 + n^2) SC: O(1)
        function rotateMatOptimal(arr: number[][]): number[][] {
          let n: number = arr.length;
          let m: number = arr[0].length;
          
          // Transpose the matrix (i.e., swapping rows with columns)
          for (let i = 0; i < n; i++) {
            for (let j = i; j < m; j++) {
              let temp: number = arr[i][j];
              arr[i][j] = arr[j][i];
              arr[j][i] = temp;
            }
          }
        
          // Reverse each row to complete the 90-degree rotation
          for (let i = 0; i < n; i++) {
            for (let j = 0; j < Math.floor(n / 2); j++) {
              let temp: number = arr[i][j];
              arr[i][j] = arr[i][n - 1 - j];
              arr[i][n - 1 - j] = temp;
            }
          }
        
          return arr;
        }
        
        ```
        
        1. kth smallest element
        
        ```jsx
        const matrix: number[][] = [
          [1, 5, 9],
          [10, 11, 13],
          [12, 13, 15],
        ];
        
        const k: number = 8;
        const result: number = kthSmallest(matrix, k);
        console.log(result); // Output: 13
        
        // TC: O(log(N) * R * log(C))
        function kthSmallest(matrix: number[][], k: number): number {
          const rows: number = matrix.length;
          const cols: number = matrix[0].length;
          let low: number = matrix[0][0];
          let high: number = matrix[rows - 1][cols - 1];
        
          while (low <= high) {
            const mid: number = low + Math.floor((high - low) / 2);
            const smallerElements: number = countSmallerElements(matrix, mid);
        
            if (smallerElements < k) {
              low = mid + 1;
            } else {
              high = mid - 1;
            }
          }
        
          return low;
        }
        
        function countSmallerElements(matrix: number[][], target: number): number {
          let count: number = 0;
        
          for (let i = 0; i < matrix.length; i++) {
            const row: number[] = matrix[i];
            let start: number = 0;
            let end: number = row.length - 1;
        
            while (start <= end) {
              const mid: number = start + Math.floor((end - start) / 2);
        
              if (row[mid] <= target) {
                start = mid + 1;
              } else {
                end = mid - 1;
              }
            }
        
            count += start;
          }
        
          return count;
        }
        
        ```
        
        1. common element
        
        ```jsx
        function commonElements(mat: number[][]): void {
          const m: number = mat.length;
          const n: number = mat[0].length;
          const common: Map<number, boolean> = new Map();
        
          // Add elements of the first row to the map
          for (let j = 0; j < n; j++) {
            common.set(mat[0][j], true);
          }
        
          // Process the remaining rows
          for (let i = 1; i < m; i++) {
            const row: number[] = mat[i];
            const currentRow: Map<number, boolean> = new Map();
        
            for (let j = 0; j < n; j++) {
              if (common.has(row[j])) {
                currentRow.set(row[j], true);
              }
            }
        
            // Clear common map and set it to current row elements
            common.clear();
            for (const [key] of currentRow) {
              common.set(key, true);
            }
          }
        
          // Print the common elements
          for (const [key] of common) {
            console.log(key + " ");
          }
        }
        
        // Example usage:
        const mat: number[][] = [
          [1, 2, 1, 4, 8],
          [3, 7, 8, 5, 1],
          [8, 7, 7, 3, 1],
          [8, 1, 2, 7, 9],
        ];
        
        commonElements(mat);
        
        ```
        
    - Heaps
        1. Max Heap
        
        ```jsx
         class BinaryHeap {
          private list: number[];
        
          constructor() {
            this.list = [];
          }
        
          // Helper method to maintain max heap property
          private maxHeapify(arr: number[], n: number, i: number): void {
            let largest = i;
            let l = 2 * i + 1;
            let r = 2 * i + 2;
        
            // Check if left child is larger than the root
            if (l < n && arr[l] > arr[largest]) {
              largest = l;
            }
        
            // Check if right child is larger than the largest so far
            if (r < n && arr[r] > arr[largest]) {
              largest = r;
            }
        
            // If largest is not root, swap and heapify the affected subtree
            if (largest !== i) {
              [arr[i], arr[largest]] = [arr[largest], arr[i]];
              this.maxHeapify(arr, n, largest);
            }
          }
        
          // Insert a new element into the heap
          insert(num: number): void {
            const size = this.list.length;
            if (size === 0) {
              this.list.push(num);
            } else {
              this.list.push(num);
              for (let i = Math.floor(this.list.length / 2) - 1; i >= 0; i--) {
                this.maxHeapify(this.list, this.list.length, i);
              }
            }
          }
        
          // Delete a specific element from the heap
          delete(num: number): void {
            let size = this.list.length;
            let i = 0;
            // Find the index of the element to delete
            for (i = 0; i < size; i++) {
              if (this.list[i] === num) {
                break;
              }
            }
        
            // Swap the element with the last element and remove the last element
            [this.list[i], this.list[size - 1]] = [this.list[size - 1], this.list[i]];
            this.list.splice(size - 1);
        
            // Re-heapify the heap after deletion
            for (let i = Math.floor(this.list.length / 2) - 1; i >= 0; i--) {
              this.maxHeapify(this.list, this.list.length, i);
            }
          }
        
          // Find the maximum element in the heap
          findMax(): number | undefined {
            return this.list[0];
          }
        
          // Delete the maximum element from the heap
          deleteMax(): void {
            if (this.list.length > 0) {
              this.delete(this.list[0]);
            }
          }
        
          // Get the size of the heap
          size(): number {
            return this.list.length;
          }
        
          // Check if the heap is empty
          isEmpty(): boolean {
            return this.list.length === 0;
          }
        
          // Get the list of elements in the heap
          getList(): number[] {
            return this.list;
          }
        }
        
        // Example usage
        const heap = new BinaryHeap();
        heap.insert(3);
        heap.insert(4);
        heap.insert(9);
        heap.insert(5);
        heap.insert(2);
        console.log(heap.getList());
        
        heap.delete(9);
        console.log(heap.getList());
        
        heap.insert(7);
        console.log(heap.getList());
        
        ```
        
        1. Min Heap
        
        ```jsx
        class BinaryHeap {
          private list: number[];
        
          constructor() {
            this.list = [];
          }
        
          // Helper method to maintain min heap property
          private minHeapify(arr: number[], n: number, i: number): void {
            let smallest = i;
            let l = 2 * i + 1;
            let r = 2 * i + 2;
        
            // Check if left child is smaller than the root
            if (l < n && arr[l] < arr[smallest]) {
              smallest = l;
            }
        
            // Check if right child is smaller than the smallest so far
            if (r < n && arr[r] < arr[smallest]) {
              smallest = r;
            }
        
            // If smallest is not root, swap and heapify the affected subtree
            if (smallest !== i) {
              [arr[i], arr[smallest]] = [arr[smallest], arr[i]];
              this.minHeapify(arr, n, smallest);
            }
          }
        
          // Insert a new element into the heap
          insert(num: number): void {
            this.list.push(num);
            let size = this.list.length;
            // Fix the heap property by performing min-heapify from the last parent node
            for (let i = Math.floor(size / 2) - 1; i >= 0; i--) {
              this.minHeapify(this.list, size, i);
            }
          }
        
          // Delete a specific element from the heap
          delete(num: number): void {
            let size = this.list.length;
            let i = 0;
            // Find the index of the element to delete
            for (i = 0; i < size; i++) {
              if (this.list[i] === num) {
                break;
              }
            }
        
            // If element was found, swap with the last element and remove it
            if (i < size) {
              [this.list[i], this.list[size - 1]] = [this.list[size - 1], this.list[i]];
              this.list.splice(size - 1);
              // Re-heapify the heap after deletion
              for (let i = Math.floor(this.list.length / 2) - 1; i >= 0; i--) {
                this.minHeapify(this.list, this.list.length, i);
              }
            }
          }
        
          // Find the minimum element in the heap
          findMin(): number | undefined {
            return this.list[0];
          }
        
          // Delete the minimum element from the heap
          deleteMin(): void {
            if (this.list.length > 0) {
              this.delete(this.list[0]);
            }
          }
        
          // Get the size of the heap
          size(): number {
            return this.list.length;
          }
        
          // Check if the heap is empty
          isEmpty(): boolean {
            return this.list.length === 0;
          }
        
          // Get the list of elements in the heap
          getList(): number[] {
            return this.list;
          }
        }
        
        // Example usage
        const heap = new BinaryHeap();
        heap.insert(3);
        heap.insert(4);
        heap.insert(9);
        heap.insert(5);
        heap.insert(2);
        
        console.log(heap.getList()); // Min-Heap after insertions
        
        heap.delete(9); // Delete element 9
        console.log(heap.getList()); // Min-Heap after deletion
        
        heap.insert(7); // Insert 7
        console.log(heap.getList()); // Min-Heap after inserting 7
        
        ```
        
    - Greedy
        1. Linked List
        
        ```jsx
        class Node {
          val: number;
          next: Node | null;
        
          constructor(val: number) {
            this.val = val;
            this.next = null;
          }
        }
        
        class LinkedList {
          head: Node | null;
        
          constructor() {
            this.head = null;
          }
        
          // Add a value to the end of the list
          add(val: number): void {
            if (this.head === null) {
              this.head = new Node(val);
              return;
            }
            let curr = this.head;
            while (curr.next !== null) {
              curr = curr.next;
            }
            curr.next = new Node(val);
          }
        
          // Add a value to the beginning of the list
          addFirst(val: number): Node | null {
            let curr = new Node(val);
            curr.next = this.head;
            this.head = curr;
            return this.head;
          }
        
          // Add a value to the end of the list (similar to add)
          addLast(val: number): void {
            let curr = this.head;
            while (curr && curr.next !== null) {
              curr = curr.next;
            }
            if (curr) {
              curr.next = new Node(val);
            } else {
              this.head = new Node(val);
            }
          }
        
          // Check if the list contains a certain value
          contains(val: number): boolean {
            let curr = this.head;
            while (curr !== null) {
              if (curr.val === val) {
                return true;
              }
              curr = curr.next;
            }
            return false;
          }
        
          // Get the value at a specific index
          get(index: number): number {
            if (index < 0) {
              throw new Error("Index out of bounds");
            }
            let curr = this.head;
            let i = 0;
            while (i < index && curr !== null) {
              curr = curr.next;
              i++;
            }
            if (curr !== null) {
              return curr.val;
            } else {
              throw new Error("Index out of bounds");
            }
          }
        
          // Peek at the value of the first node
          peek(): number {
            if (this.head === null) {
              throw new Error("List is empty");
            }
            return this.head.val;
          }
        
          // Remove and return the value of the first node
          poll(): number {
            if (this.head === null) {
              throw new Error("List is empty");
            }
            let value = this.head.val;
            this.head = this.head.next;
            return value;
          }
        
          // Pop (remove) the first node (alias for poll)
          pop(): number {
            return this.poll();
          }
        
          // Push a new value at the beginning of the list (alias for addFirst)
          push(element: number): Node | null {
            return this.addFirst(element);
          }
        
          // Remove a node at a specific index
          remove(index: number): void {
            if (index < 0) {
              throw new Error("Index out of bounds");
            }
            if (index === 0) {
              if (this.head !== null) {
                this.head = this.head.next;
                return;
              } else {
                throw new Error("Index out of bounds");
              }
            }
            let curr = this.head;
            let prev: Node | null = null;
            let i = 0;
            while (i < index && curr !== null) {
              prev = curr;
              curr = curr.next;
              i++;
            }
        
            if (curr !== null) {
              if (prev !== null) {
                prev.next = curr.next;
              }
            } else {
              throw new Error("Index out of bounds");
            }
          }
        
          // Get the size of the list
          size(): number {
            let count = 0;
            let curr = this.head;
            while (curr !== null) {
              count++;
              curr = curr.next;
            }
            return count;
          }
        
          // Create a cycle in the linked list (for testing)
          createCircle(startIndex: number, cycleIndex: number): void {
            if (this.head === null) {
              throw new Error("Cannot create a cycle in an empty list.");
            }
        
            let currIndex = 0;
            let currNode: Node | null = this.head;
            let startNode: Node | null = null;
            let endNode: Node | null = null;
        
            while (currNode !== null) {
              if (currIndex === startIndex) {
                startNode = currNode;
              }
              if (currIndex === cycleIndex) {
                endNode = currNode;
              }
              currNode = currNode.next;
              currIndex++;
            }
        
            if (startNode === null || endNode === null) {
              throw new Error("Invalid cycle position or start index.");
            }
        
            endNode.next = startNode;
          }
        
          // Print the list, detecting cycles to avoid infinite loops
          print(): void {
            let curr = this.head;
            let visitedNodes = new Set<Node>();
            let str = "";
        
            while (curr !== null) {
              if (visitedNodes.has(curr)) {
                console.log("Cycle detected. Stopping print.");
                break;
              }
              visitedNodes.add(curr);
              str += curr.val + "->";
              curr = curr.next;
            }
        
            if (curr === null) {
              console.log(str + "null (end of list)");
            }
          }
        }
        
        // Example usage
        const list = new LinkedList();
        list.add(10);
        list.add(20);
        list.add(30);
        list.addFirst(5);
        list.addLast(40);
        list.print(); // 5->10->20->30->40->null
        
        list.remove(2);
        list.print(); // 5->10->30->40->null
        
        console.log(list.get(2)); // 30
        
        ```
        
        1. Find Middle Node
        
        ```jsx
        import { LinkedList } from "./01.linked-list"; // Assuming the LinkedList and Node classes are exported correctly from the other file
        
        function middleNode(list: LinkedList): Node | null {
          if (list.head === null) {
            return null; // Empty list, no middle node
          }
        
          let slow: Node | null = list.head;
          let fast: Node | null = list.head;
        
          while (fast !== null && fast.next !== null) {
            slow = slow.next;
            fast = fast.next.next;
          }
        
          return slow;
        }
        
        const list = new LinkedList();
        list.add("a");
        list.add("b");
        list.add("c");
        list.add("d");
        list.add("e");
        list.add("f");
        list.createCircle(1, 2);
        list.print();
        
        const middle = middleNode(list);
        if (middle !== null) {
          console.log("Middle node:", middle.val);
        } else {
          console.log("The list is empty.");
        }
        
        ```
        
        1. Reverse LinkedList
        
        ```jsx
        import { LinkedList, Node } from "./01.linked-list"; // Assuming LinkedList and Node are properly exported
        
        // Create an instance of LinkedList
        const list = new LinkedList();
        list.add("a");
        list.add("b");
        list.add("c");
        list.add("d");
        list.add("e");
        list.add("f");
        
        // Iterative Reverse Function
        function reverse(list: LinkedList): void {
          let prev: Node | null = null;
          let curr: Node | null = list.head;
          let next: Node | null = null;
        
          while (curr !== null) {
            next = curr.next;
            curr.next = prev;
            prev = curr;
            curr = next;
          }
        
          list.head = prev;
        }
        
        // Recursive Reverse Function
        function reverseList(node: Node | null): Node | null {
          if (node === null || node.next === null) {
            return node;
          }
        
          let newHead = reverseList(node.next);
          node.next.next = node;
          node.next = null;
          
          return newHead;
        }
        
        // Reverse the list using the iterative method
        reverse(list);
        
        // Print the reversed list
        list.print();
        
        // Reverse the list using the recursive method and update the head of the list
        const newHead = reverseList(list.head);
        list.head = newHead;
        
        // Print the recursively reversed list
        list.print();
        
        ```
        
        1. Detect Cycle
        
        ```jsx
        import { LinkedList, Node } from "./01.linked-list"; // Assuming LinkedList and Node are properly exported
        
        const list = new LinkedList();
        list.add(1);
        list.add(2);
        list.add(3);
        list.add(4);
        list.add(5);
        list.add(6);
        list.createCircle(1, 5); // Creates a cycle between node 1 and 5
        list.print();
        
        // Check for cycle using the optimal approach
        console.log("Cycle detected:", detectCycleOptimal(list.head));
        
        // Naive cycle detection using a set
        function detectCycleNaive(head: Node | null): boolean {
          const set = new Set<Node>();
          while (head !== null) {
            if (set.has(head)) return true; // Checking the node reference directly
            set.add(head);
            head = head.next;
          }
          return false;
        }
        
        // Optimal cycle detection using slow and fast pointers
        function detectCycleOptimal(head: Node | null): boolean {
          if (head === null || head.next === null) return false;
        
          let slow: Node | null = head;
          let fast: Node | null = head;
        
          while (fast !== null && fast.next !== null) {
            fast = fast.next.next;  // Move fast pointer two steps
            slow = slow.next;       // Move slow pointer one step
        
            if (fast === slow) {
              return true; // Cycle detected when slow and fast pointers meet
            }
          }
          return false; // No cycle detected
        }
        
        ```
        
        5.Find Starting Index
        
        ```jsx
        import { LinkedList, Node } from "./01.linked-list"; // Assuming LinkedList and Node are correctly exported
        
        // Create the linked list
        const list = new LinkedList();
        list.add(1);
        list.add(2);
        list.add(3);
        list.add(4);
        list.add(5);
        list.add(6);
        list.createCircle(1, 5); // Creates a cycle between node 1 and 5
        list.print();
        
        // Check for cycle using the optimal approach
        console.log("ans:", detectCycleOptimal(list.head));
        
        // Naive cycle detection (returns the node's value where the cycle starts)
        function detectCycleNaive(head: Node | null): number | null {
          const set = new Set<Node>();
          while (head !== null) {
            if (set.has(head)) return head.val; // If a node is revisited, return its value (cycle start)
            set.add(head);
            head = head.next;
          }
          return null; // No cycle detected
        }
        
        // Optimal cycle detection (returns the node's value where the cycle starts)
        function detectCycleOptimal(head: Node | null): number | null {
          let fast: Node | null = head;
          let slow: Node | null = head;
          let entry: Node | null = head;
        
          while (fast !== null && fast.next !== null) {
            fast = fast.next.next;
            slow = slow.next;
        
            // Cycle detected
            if (slow === fast) {
              // Move slow pointer back to the head
              while (slow !== entry) {
                slow = slow.next;
                entry = entry.next;
              }
              return slow.val; // Return the node's value where the cycle starts
            }
          }
        
          return null; // No cycle detected
        }
        
        ```
        
        1. Length of Loop
        
        ```jsx
        import { LinkedList, Node } from "./01.linked-list"; // Assuming LinkedList and Node are exported from the module
        
        // Create the linked list and add nodes
        const list = new LinkedList();
        list.add(1);
        list.add(2);
        list.add(3);
        list.add(4);
        list.add(5);
        list.add(6);
        list.createCircle(2, 5); // Creates a cycle between nodes 2 and 5
        list.print();
        
        // Check for cycle length using the optimal approach
        console.log("Cycle Length: ", lengthOfLoop(list.head));
        
        // Naive approach to calculate the length of the loop in the linked list
        function lengthOfLoopNaive(head: Node | null): number | null {
          const set = new Set<Node>();
          let totalLength = 0;
          let length = 0;
          let entry = head;
        
          while (head !== null) {
            if (set.has(head)) {
              // Cycle detected, count the length of the cycle
              while (entry !== head) {
                entry = entry.next;
                length++;
              }
              return totalLength - length; // Return the length of the cycle
            }
            set.add(head);
            totalLength++;
            head = head.next;
          }
        
          return null; // No cycle detected
        }
        
        // Optimal approach (Floyd's Cycle-Finding Algorithm) to calculate the length of the loop
        function lengthOfLoop(head: Node | null): number | false {
          let slow: Node | null = head;
          let fast: Node | null = head;
          let length = 0;
        
          // Detect cycle using slow and fast pointers
          while (fast !== null && fast.next !== null) {
            fast = fast.next.next;
            slow = slow.next;
        
            if (slow === fast) {
              // Cycle detected, now calculate the length of the cycle
              do {
                length++;
                fast = fast.next;
              } while (slow !== fast);
        
              return length; // Return the length of the cycle
            }
          }
        
          return false; // No cycle detected
        }
        
        ```
        
        7, Check If LL Palindrome
        
        ```jsx
        // Assuming LinkedList and Node are defined in the linked list module
        import { LinkedList, Node } from "./01.linked-list";
        
        const list = new LinkedList();
        list.add(1);
        list.add(2);
        list.add(3);
        list.add(3);
        list.add(2);
        list.add(1);
        list.print();
        
        // Check if the linked list is a palindrome
        console.log("Ans: ", isPalindrome(list.head));
        
        function isPalindromeNaive(head: Node | null): boolean {
          let ans: number[] = [];
          while (head !== null) {
            ans.push(head.val);
            head = head.next;
          }
        
          let n = ans.length;
          for (let i = 0; i < Math.floor(n / 2); i++) {
            if (ans[i] !== ans[n - 1 - i]) return false;
          }
          return true;
        }
        
        function isPalindrome(head: Node | null): boolean {
          if (head == null || head.next == null) return true;
        
          let slow: Node | null = head;
          let fast: Node | null = head;
        
          // Move slow to the middle of the list
          while (fast !== null && fast.next !== null) {
            slow = slow.next;
            fast = fast.next.next;
          }
        
          // Reverse the second half of the list
          slow.next = reverse(slow.next);
          slow = slow.next;
        
          let temp: Node | null = head;
          while (slow !== null) {
            if (temp!.val !== slow.val) return false;
            slow = slow.next;
            temp = temp.next;
          }
        
          return true;
        }
        
        function reverse(head: Node | null): Node | null {
          let prev: Node | null = null;
          let curr: Node | null = head;
          let next: Node | null = null;
        
          while (curr !== null) {
            next = curr.next;
            curr.next = prev;
            prev = curr;
            curr = next;
          }
        
          return prev;
        }
        
        ```
        
        1. Odd Even LL
        
        ```jsx
        // Assuming LinkedList and Node are defined in the linked list module
        import { LinkedList, Node } from "./01.linked-list";
        
        const list = new LinkedList();
        list.add(2);
        list.add(1);
        list.add(3);
        list.add(5);
        list.add(6);
        list.add(4);
        list.add(7);
        
        console.log("Original List:");
        list.print();
        
        // Call the function to rearrange the list (odd-even positions)
        console.log(oddEvenNaive(list.head));
        
        function oddEvenNaive(head: Node | null): Node | null {
          if (head == null) return null;
        
          let odd: Node | null = head;
          let even: Node | null = head.next;
          let evenHead: Node | null = even;
        
          while (even !== null && even.next !== null) {
            odd.next = odd.next?.next ?? null;  // Move odd pointer to next odd node
            odd = odd.next ?? null;
        
            even.next = even.next?.next ?? null;  // Move even pointer to next even node
            even = even.next ?? null;
          }
        
          // Connect the end of the odd list to the head of the even list
          if (odd !== null) {
            odd.next = evenHead;
          }
        
          return head;
        }
        
        ```
 
