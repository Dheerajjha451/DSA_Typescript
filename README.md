- Typescript
    
    # Array

    
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
    
    2. minMaxArray
    
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
    
    3. Kth smallest
    
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
    
    4. Sort (0,1,2)
    
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
    
    5. Move Negative left
    
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
    
    6. Find Union and InterSection
    
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
    
    7. Cyclic Rotate Array
    
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
    
    8. Sub Array Kadane
    
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
    
    9. Minimise Height
    
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
    
    10. Minimum Number of Jump
    
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
    
    11. Find Duplicates
    
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
    
    12. Merge 
    
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
    
    13. MergeIntervals
    
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
    
    14. Next Permutation
    
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
    
    15. Count Inversion
    
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
    
    16. Stock Buy And Sell
    
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
    
    18. Common Elements 3 sorted Array
    
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
    
    19. Rearrange by Sign
    
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
    
    20. Sub Array k0
    
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
    
    21. Max Product Sub Array
    
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
    
    22. Longes Consecutive Sucessive
    
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
    
    23. Element Appear NK times
    
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
    
    24. Max Profit 2 Data
    
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
    
    25. Subset of Other Array
    
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
    
    26. Three Sum
    
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
    
    27. Trapping RainWater
    
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
    
    28. Factorial of large number
    
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
        
        2. Lower Bound
        
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
        
        3. Upper Bound
        
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
        
        4. Search Insert Position
        
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
        
        5. Floor Ceil
        
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
        
        6. First Last Occurance
        
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
        
        2. For Each
        
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
        
        3. Map
        
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
        
        4. Reduce
        
        ```tsx
        // Adding myOwnReduce method to Array prototype
        Array.prototype.myOwnReduce = function <T>(callback: (accumulator: T, currentValue: T, index?: number, array?: T[]) => T, initialValue?: T): T {
          if (this.length === 0 && initialValue === undefined) {
            throw new TypeError("Reduce of empty array with no initial value");
          }
        
          let accumulator: T = initialValue !== undefined ? initialValue : this[0];
          const startIndex = initialValue !== undefined ? 0 : 1;
        
          for (let i = startIndex; i < this.length; i++) {
            accumulator = callback(accumulator, this[i], i, this);
          }
        
          return accumulator;
        };
        
        const sum = arr.myOwnReduce((accumulator, currentValue) => {
          return accumulator + currentValue;
        }, 0);
        
        console.log("Reduced Sum:", sum);
        
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
        
        2. Max Sub Array Size K
        
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
        
        3. First Negative Number of Size K
        
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
        
        2. Reverse String
        
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
        
        3. Reverse Word
        
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
        
        4. Duplicate Char String
        
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
        
        5. Odd Number In String
        
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
        
        6. Longest Common Prefix
        
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
        
        7. Palindrome
        
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
        
        8. Isomorphic String
        
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
        
        9. Check if Valid Anagram
        
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
        
        10. Sort Char Freq
        
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
        
        11. Max Depth Parenthesis
        
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
        
        12. Roman To Integer
        
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
        
        13. Integer to Roman
        
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
        
        14. Count Distinct Character
        
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
        
        15. Longest Palindrome
        
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
        
        16. Sum of beauty
        
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
        
        17. Atoi
        
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
        
        2. Stack Using LinkedList
        
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
        
        3. Queue Using ArrayList
        
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
        
        4. Queue using LinkedList
        
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
        
        5. Valid Paranthesis
        
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
        
        6. Stack Using Queue
        
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
        
        7. Queue Using Stack
        
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
        
        8. Min Stack
        
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
        
        9. Next Greater Element
        
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
        
        10. Next Smallest Element
        
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
        
        11. Trapping Rain Water
        
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
        
        12. Sum of Sub Array Min
        
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
        
        13. Histogram
        
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
        
        14. Asteroid Collision
        
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
        
        15. Sum of Sub Arrays Ranges
        
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
        
        16. Remove K Digits
        
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
        
        17. LRU Cache
        
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
