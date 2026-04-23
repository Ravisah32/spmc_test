import java.io.*;
import java.util.*;

public class E1_optimized {

    public static void main(String[] args) throws Exception {
        if (args.length < 2) {
            System.out.println("Usage: java E <input.csv> <number_of_colors>");
            return;
        }

        final String csvFile = args[0];
        final int numberOfColors = Integer.parseInt(args[1]);

        long t0 = System.nanoTime();
        Data data = readCSV(csvFile);
        if (data.m == 0) {
            System.out.println("No pairwise columns like 'Red-Blue' found in header.");
            return;
        }
        if (data.uniqueColors.size() != numberOfColors) {
            System.out.println("number_of_colors mismatch: file has " + data.uniqueColors.size() +
                    " unique colors but arg was " + numberOfColors);
            return;
        }

        // The last dimension holds the row index t, so we need m+1 total dims
        RangeTree tree = new RangeTree(data.points, data.m + 1);
        long t1 = System.nanoTime();
        System.out.println("Tree built in " + ((t1 - t0) / 1_000_000) + " ms");

        BufferedReader br = new BufferedReader(new InputStreamReader(System.in));
        while (true) {
            System.out.print("Enter x and range [start end] (enter -1 -1 -1 to exit):\n");
            String line = br.readLine();
            if (line == null)
                break;
            line = line.trim();
            if (line.isEmpty())
                continue;
            String[] parts = line.split("\\s+");
            if (parts.length < 3)
                continue;
            int epsilon;
            double iStart = 0, iEnd = 0;
            try {
                epsilon = Integer.parseInt(parts[0]);
                iStart = Double.parseDouble(parts[1]);
                iEnd = Double.parseDouble(parts[2]);
            } catch (Exception ex) {
                continue;
            }
            if (epsilon == -1 && iStart == -1 && iEnd == -1)
                break;

            long q0 = System.nanoTime();
            Result res = queryBest(data, tree, epsilon, iStart, iEnd);
            long q1 = System.nanoTime();

            if (res == null) {
                System.out.println("Best range = [NA,NA] (similarity = 0.0)");
            } else {
                System.out.println("Best range = [" + res.leftVal + "," + res.rightVal +
                        "] (similarity = " + res.similarity + ")");
            }
            System.out.println("Query executed in " + ((q1 - q0) / 1_000_000) + " ms");
        }
    }

    // Just a small container to pass query results back cleanly
    static final class Result {
        final double leftVal, rightVal, similarity;

        Result(double l, double r, double s) {
            leftVal = l;
            rightVal = r;
            similarity = s;
        }
    }

    static Result queryBest(Data data, RangeTree tree, int epsilon, double startVal, double endVal) {
        int n = data.n;
        if (n == 0)
            return null;

        // Translate the value-space endpoints into index positions
        int L0 = lowerBound(data.keys, Math.min(startVal, endVal));
        int R0 = upperBound(data.keys, Math.max(startVal, endVal)) - 1;
        if (L0 < 0)
            L0 = 0;
        if (R0 >= n)
            R0 = n - 1;
        if (L0 > R0)
            return null;

        // Switch to 1-based so the Jaccard math stays simple
        int qL = L0 + 1;
        int qR = R0 + 1;

        // The key insight: only a handful of R values near qL and qR can possibly
        // maximize Jaccard — no need to scan every R in [p+1, n].
        // We just probe the immediate neighbors of both endpoints.
        Set<Integer> candidateRSet = new LinkedHashSet<>();
        if (qL - 1 >= 1)
            candidateRSet.add(qL - 1);
        if (qL <= n)
            candidateRSet.add(qL);
        if (qL + 1 <= n)
            candidateRSet.add(qL + 1);
        if (qR - 1 >= 1)
            candidateRSet.add(qR - 1);
        if (qR <= n)
            candidateRSet.add(qR);
        if (qR + 1 <= n)
            candidateRSet.add(qR + 1);

        int[] candidateRs = candidateRSet.stream().mapToInt(Integer::intValue).toArray();

        double bestJ = -1.0;
        int bestL = -1, bestR = -1;

        // Try every possible left boundary and pair it with each candidate R
        for (int p = 0; p < n; p++) {
            double[] center = data.points[p];

            double[] lo = new double[data.m + 1];
            double[] hi = new double[data.m + 1];

            // Build the epsilon-ball around this point in color space
            for (int d = 0; d < data.m; d++) {
                lo[d] = center[d] - epsilon;
                hi[d] = center[d] + epsilon;
            }

            for (int candR : candidateRs) {
                if (candR <= p)
                    continue;

                // Pin the index dimension to exactly this R value and check existence
                lo[data.m] = candR;
                hi[data.m] = candR;

                boolean found = tree.pointExistsInRange(lo, hi);
                if (!found)
                    continue;

                int L = p + 1;
                int R = candR;

                double j = jaccard(L, R, qL, qR);
                if (j > bestJ) {
                    bestJ = j;
                    bestL = L;
                    bestR = R;
                } else if (j == bestJ && j >= 0) {
                    // Break ties: prefer the shorter range, then leftmost, then smallest R
                    int wBest = bestR - bestL, wCur = R - L;
                    if (wCur < wBest || (wCur == wBest &&
                            (L < bestL || (L == bestL && R < bestR)))) {
                        bestL = L;
                        bestR = R;
                    }
                }
            }
        }

        if (bestL == -1)
            return null;
        double leftVal = data.keys[bestL - 1];
        double rightVal = data.keys[bestR - 1];
        return new Result(leftVal, rightVal, bestJ);
    }

    static final class Data {
        final int n;
        final double[] keys;
        final int m;
        final List<String> pairNames;
        final Set<String> uniqueColors;
        final int[][] prefix;
        final double[][] points; // (m+1)-dimensional: color prefix sums + row index

        Data(int n, double[] keys, int m, List<String> pairNames, Set<String> uniqueColors,
                int[][] prefix, double[][] points) {
            this.n = n;
            this.keys = keys;
            this.m = m;
            this.pairNames = pairNames;
            this.uniqueColors = uniqueColors;
            this.prefix = prefix;
            this.points = points;
        }
    }

    static Data readCSV(String filename) throws IOException {
        try (BufferedReader br = new BufferedReader(new FileReader(filename))) {
            String header = br.readLine();
            if (header == null)
                throw new IllegalArgumentException("Empty CSV");
            String[] cols = splitCSV(header);

            Integer indexCol = null;
            List<Integer> pairCols = new ArrayList<>();
            List<String> pairNames = new ArrayList<>();
            Set<String> colors = new LinkedHashSet<>();

            // Figure out which columns are pair columns (e.g. "Red-Blue") vs the index
            for (int i = 0; i < cols.length; i++) {
                String name = cols[i].trim();
                if (name.equalsIgnoreCase("index"))
                    indexCol = i;
                else if (isPairName(name)) {
                    pairCols.add(i);
                    pairNames.add(name);
                    String[] ab = name.split("-");
                    if (ab.length == 2) {
                        colors.add(ab[0]);
                        colors.add(ab[1]);
                    }
                }
            }
            if (pairCols.isEmpty())
                return new Data(0, new double[0], 0, pairNames, colors, new int[0][0], new double[0][0]);

            List<double[]> rows = new ArrayList<>();
            List<Double> key = new ArrayList<>();
            String line;
            while ((line = br.readLine()) != null) {
                if (line.isEmpty())
                    continue;
                String[] parts = splitCSV(line);
                double idxVal;
                if (indexCol != null && indexCol < parts.length)
                    idxVal = parseDoubleSafe(parts[indexCol]);
                else {
                    // Fall back to the first column, or just use row number if that fails too
                    Double maybe = tryParse(parts[0]);
                    idxVal = (maybe != null) ? maybe : (key.size() + 1);
                }
                key.add(idxVal);
                double[] vals = new double[pairCols.size()];
                for (int k = 0; k < pairCols.size(); k++) {
                    int cIdx = pairCols.get(k);
                    vals[k] = (cIdx < parts.length && !parts[cIdx].isEmpty())
                            ? parseDoubleSafe(parts[cIdx])
                            : 0.0;
                }
                rows.add(vals);
            }

            int n = rows.size();
            int m = pairCols.size();
            double[] keys = new double[n];
            for (int i = 0; i < n; i++)
                keys[i] = key.get(i);

            // Build prefix sums per color-pair dimension
            int[][] prefix = new int[m][n + 1];
            for (int t = 0; t < n; t++) {
                double[] row = rows.get(t);
                for (int d = 0; d < m; d++)
                    prefix[d][t + 1] = (int) Math.round(row[d]);
            }

            // Each point is the prefix-sum vector at position t, with t appended as its own
            // dimension
            double[][] points = new double[n + 1][m + 1];
            for (int t = 0; t <= n; t++) {
                for (int d = 0; d < m; d++)
                    points[t][d] = (t == 0) ? 0 : prefix[d][t];
                points[t][m] = t;
            }

            return new Data(n, keys, m, pairNames, colors, prefix, points);
        }
    }

    // A valid pair name looks like "Red-Blue" — exactly one dash, not at either end
    static boolean isPairName(String name) {
        int dash = name.indexOf('-');
        return dash > 0 && dash < name.length() - 1 && name.indexOf('-', dash + 1) == -1;
    }

    static String[] splitCSV(String line) {
        List<String> out = new ArrayList<>();
        StringBuilder sb = new StringBuilder();
        boolean inQ = false;
        for (int i = 0; i < line.length(); i++) {
            char ch = line.charAt(i);
            if (ch == '"')
                inQ = !inQ;
            else if (ch == ',' && !inQ) {
                out.add(sb.toString());
                sb.setLength(0);
            } else
                sb.append(ch);
        }
        out.add(sb.toString());
        return out.toArray(new String[0]);
    }

    static Double tryParse(String s) {
        try {
            return Double.parseDouble(s.trim());
        } catch (Exception e) {
            return null;
        }
    }

    static double parseDoubleSafe(String s) {
        try {
            return Double.parseDouble(s.trim());
        } catch (Exception e) {
            return 0.0;
        }
    }

    // A basic KD-tree style range tree. Nothing fancy, but it gets the job done.
    static final class RangeTree {
        final int dims;
        final Node root;
        final double[][] P;

        RangeTree(double[][] points, int dims) {
            this.dims = dims;
            this.P = points;
            int n = points.length;
            int[] all = new int[n];
            for (int i = 0; i < n; i++)
                all[i] = i;
            this.root = build(all, 0, n, 0);
        }

        final class Node {
            double[] lo, hi;
            int dim;
            double split;
            Node left, right;
            int[] idxs;
            boolean isLeaf;
        }

        Node build(int[] idx, int loi, int hii, int depth) {
            int n = hii - loi;
            Node node = new Node();
            node.lo = new double[dims];
            node.hi = new double[dims];
            Arrays.fill(node.lo, Double.POSITIVE_INFINITY);
            Arrays.fill(node.hi, Double.NEGATIVE_INFINITY);

            // Compute the bounding box for this node's points
            for (int k = loi; k < hii; k++) {
                double[] pt = P[idx[k]];
                for (int d = 0; d < dims; d++) {
                    if (pt[d] < node.lo[d])
                        node.lo[d] = pt[d];
                    if (pt[d] > node.hi[d])
                        node.hi[d] = pt[d];
                }
            }

            // Small enough to just store as a leaf and scan linearly
            if (n <= 32) {
                node.isLeaf = true;
                node.idxs = Arrays.copyOfRange(idx, loi, hii);
                return node;
            }

            // Cycle through dimensions as we go deeper
            int dim = depth % dims;
            int mid = loi + n / 2;
            nthElement(idx, loi, mid, hii, dim);
            node.dim = dim;
            node.split = P[idx[mid]][dim];
            node.left = build(idx, loi, mid, depth + 1);
            node.right = build(idx, mid, hii, depth + 1);
            return node;
        }

        // Partial quickselect — rearranges so idx[mid] ends up in its sorted position
        void nthElement(int[] a, int lo, int mid, int hi, int dim) {
            int l = lo, r = hi - 1;
            while (true) {
                int i = l, j = r;
                double pivot = P[a[(l + r) >>> 1]][dim];
                while (i <= j) {
                    while (P[a[i]][dim] < pivot)
                        i++;
                    while (P[a[j]][dim] > pivot)
                        j--;
                    if (i <= j) {
                        int t = a[i];
                        a[i] = a[j];
                        a[j] = t;
                        i++;
                        j--;
                    }
                }
                if (j < mid)
                    l = i;
                else
                    r = j;
                if (l >= mid && r <= mid)
                    return;
            }
        }

        // We only need to know if at least one point falls in the box — no need to
        // collect them all
        boolean pointExistsInRange(double[] lo, double[] hi) {
            return pointExistsInRange(root, lo, hi);
        }

        boolean pointExistsInRange(Node node, double[] lo, double[] hi) {
            if (node == null)
                return false;
            if (!overlaps(node, lo, hi))
                return false;
            if (node.isLeaf) {
                for (int id : node.idxs)
                    if (contains(P[id], lo, hi))
                        return true;
                return false;
            }
            return pointExistsInRange(node.left, lo, hi) ||
                    pointExistsInRange(node.right, lo, hi);
        }

        // Quick bounding-box rejection test
        boolean overlaps(Node node, double[] qlo, double[] qhi) {
            for (int d = 0; d < dims; d++)
                if (node.hi[d] < qlo[d] || node.lo[d] > qhi[d])
                    return false;
            return true;
        }

        boolean contains(double[] pt, double[] qlo, double[] qhi) {
            for (int d = 0; d < dims; d++)
                if (pt[d] < qlo[d] || pt[d] > qhi[d])
                    return false;
            return true;
        }
    }

    // Standard binary search — returns first index where a[i] >= x
    static int lowerBound(double[] a, double x) {
        int lo = 0, hi = a.length;
        while (lo < hi) {
            int mid = (lo + hi) >>> 1;
            if (a[mid] < x)
                lo = mid + 1;
            else
                hi = mid;
        }
        return lo;
    }

    // Returns first index where a[i] > x
    static int upperBound(double[] a, double x) {
        int lo = 0, hi = a.length;
        while (lo < hi) {
            int mid = (lo + hi) >>> 1;
            if (a[mid] <= x)
                lo = mid + 1;
            else
                hi = mid;
        }
        return lo;
    }

    static double jaccard(int aL, int aR, int bL, int bR) {
        int inter = intersectionSize(aL, aR, bL, bR);
        int uni = (aR - aL + 1) + (bR - bL + 1) - inter;
        return (uni == 0) ? 0.0 : ((double) inter) / uni;
    }

    static int intersectionSize(int aL, int aR, int bL, int bR) {
        int L = Math.max(aL, bL), R = Math.min(aR, bR);
        return (L <= R) ? (R - L + 1) : 0;
    }
}