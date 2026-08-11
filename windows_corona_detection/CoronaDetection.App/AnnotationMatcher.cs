namespace CoronaDetection;

internal static class AnnotationMatcher
{
    public static IReadOnlyList<AnnotationMatch> Match(
        IReadOnlyList<AnnotationBox> groundTruth,
        IReadOnlyList<AnnotationBox> predictions,
        double iouThreshold)
    {
        if (groundTruth.Count == 0 || predictions.Count == 0) return [];
        int size = Math.Max(groundTruth.Count, predictions.Count);
        var weights = new double[size, size];
        var ious = new double[groundTruth.Count, predictions.Count];
        for (int gt = 0; gt < groundTruth.Count; gt++)
        {
            for (int pred = 0; pred < predictions.Count; pred++)
            {
                double iou = CalculateIou(groundTruth[gt].Box, predictions[pred].Box);
                ious[gt, pred] = iou;
                if (iou >= iouThreshold)
                    weights[gt, pred] = 2.0 + iou;
            }
        }

        int[] assignment = MaximizeAssignment(weights);
        var matches = new List<AnnotationMatch>();
        for (int gt = 0; gt < groundTruth.Count; gt++)
        {
            int pred = assignment[gt];
            if (pred < 0 || pred >= predictions.Count || ious[gt, pred] < iouThreshold)
                continue;
            matches.Add(new AnnotationMatch(
                groundTruth[gt],
                predictions[pred],
                ious[gt, pred],
                groundTruth[gt].Label.Equals(
                    predictions[pred].Label, StringComparison.OrdinalIgnoreCase)));
        }
        return matches;
    }

    public static double CalculateIou(BoundingBox left, BoundingBox right)
    {
        double intersectionWidth = Math.Max(0.0, Math.Min(left.Right, right.Right) - Math.Max(left.X, right.X));
        double intersectionHeight = Math.Max(0.0, Math.Min(left.Bottom, right.Bottom) - Math.Max(left.Y, right.Y));
        double intersection = intersectionWidth * intersectionHeight;
        double union = left.Width * left.Height + right.Width * right.Height - intersection;
        return union > 0.0 ? intersection / union : 0.0;
    }

    // Hungarian assignment. Candidate weights include a cardinality bonus, so
    // the solution first maximizes the number of IoU-qualified matches and then
    // their total IoU.
    private static int[] MaximizeAssignment(double[,] weights)
    {
        int size = weights.GetLength(0);
        var u = new double[size + 1];
        var v = new double[size + 1];
        var p = new int[size + 1];
        var way = new int[size + 1];
        const double maxWeight = 3.0;

        for (int row = 1; row <= size; row++)
        {
            p[0] = row;
            int column0 = 0;
            var minimum = Enumerable.Repeat(double.PositiveInfinity, size + 1).ToArray();
            var used = new bool[size + 1];
            do
            {
                used[column0] = true;
                int currentRow = p[column0];
                double delta = double.PositiveInfinity;
                int column1 = 0;
                for (int column = 1; column <= size; column++)
                {
                    if (used[column]) continue;
                    double cost = maxWeight - weights[currentRow - 1, column - 1];
                    double current = cost - u[currentRow] - v[column];
                    if (current < minimum[column])
                    {
                        minimum[column] = current;
                        way[column] = column0;
                    }
                    if (minimum[column] < delta)
                    {
                        delta = minimum[column];
                        column1 = column;
                    }
                }
                for (int column = 0; column <= size; column++)
                {
                    if (used[column])
                    {
                        u[p[column]] += delta;
                        v[column] -= delta;
                    }
                    else
                    {
                        minimum[column] -= delta;
                    }
                }
                column0 = column1;
            } while (p[column0] != 0);

            do
            {
                int column1 = way[column0];
                p[column0] = p[column1];
                column0 = column1;
            } while (column0 != 0);
        }

        var assignment = Enumerable.Repeat(-1, size).ToArray();
        for (int column = 1; column <= size; column++)
            if (p[column] > 0) assignment[p[column] - 1] = column - 1;
        return assignment;
    }
}
