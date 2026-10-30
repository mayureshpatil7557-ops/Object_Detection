"""
tracker.py — Python Centroid Tracker
Assigns persistent IDs across frames (SORT-inspired)
"""

import math


class CentroidTracker:
    def __init__(self, max_disappeared: int = 25, max_distance: int = 100):
        self.next_id        = 0
        self.objects        = {}   # id → {centroid, label, color, age}
        self.disappeared    = {}   # id → frame count
        self.max_disappeared = max_disappeared
        self.max_distance    = max_distance

    def _dist(self, a, b):
        return math.sqrt((a['x'] - b['x'])**2 + (a['y'] - b['y'])**2)

    def _register(self, centroid, label, color):
        self.objects[self.next_id] = {
            'centroid': centroid,
            'label':    label,
            'color':    color,
            'age':      0,
        }
        self.disappeared[self.next_id] = 0
        self.next_id += 1

    def _deregister(self, oid):
        del self.objects[oid]
        del self.disappeared[oid]

    def update(self, detections: list) -> dict:
        """
        detections: list of { label, score, bbox:[x,y,w,h], color }
        returns:    dict of { id: {centroid, label, color, age} }
        """
        if not detections:
            for oid in list(self.disappeared.keys()):
                self.disappeared[oid] += 1
                if self.disappeared[oid] > self.max_disappeared:
                    self._deregister(oid)
            return dict(self.objects)

        # Compute centroids for this frame's detections
        input_cents = []
        for d in detections:
            x, y, w, h = d['bbox']
            input_cents.append({
                'x':     x + w / 2,
                'y':     y + h / 2,
                'label': d['label'],
                'color': d['color'],
            })

        if not self.objects:
            for c in input_cents:
                self._register(c, c['label'], c['color'])
            return dict(self.objects)

        obj_ids   = list(self.objects.keys())
        obj_cents = [self.objects[oid]['centroid'] for oid in obj_ids]

        # Distance matrix
        D = [
            [self._dist(oc, ic) for ic in input_cents]
            for oc in obj_cents
        ]

        # Greedy matching (sort rows by min distance)
        rows_sorted = sorted(
            range(len(D)), key=lambda r: min(D[r]) if D[r] else float('inf')
        )

        used_rows, used_cols = set(), set()

        for r in rows_sorted:
            if r in used_rows or not D[r]:
                continue
            row  = D[r]
            c    = min(range(len(row)), key=lambda i: row[i])
            if c in used_cols:
                continue
            if row[c] > self.max_distance:
                continue

            oid = obj_ids[r]
            ic  = input_cents[c]
            self.objects[oid]['centroid'] = ic
            self.objects[oid]['label']    = ic['label']
            self.objects[oid]['color']    = ic['color']
            self.objects[oid]['age']     += 1
            self.disappeared[oid]         = 0
            used_rows.add(r)
            used_cols.add(c)

        # Register unmatched inputs
        for c, ic in enumerate(input_cents):
            if c not in used_cols:
                self._register(ic, ic['label'], ic['color'])

        # Increment disappeared for unmatched existing objects
        for r, oid in enumerate(obj_ids):
            if r not in used_rows:
                self.disappeared[oid] += 1
                if self.disappeared[oid] > self.max_disappeared:
                    self._deregister(oid)

        return dict(self.objects)

    def reset(self):
        self.objects.clear()
        self.disappeared.clear()
        self.next_id = 0
