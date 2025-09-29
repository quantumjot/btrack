from __future__ import annotations

from btrack import btypes

import csv
import os


def import_CSV(filename: os.PathLike) -> list[btypes.PyTrackObject]:
    """Import localizations from a CSV file.

    Parameters
    ----------
    filename : PathLike
        The filename of the CSV to import

    Returns
    -------
    objects : list[btypes.PyTrackObject]
        A list of objects in the CSV file.

    Notes
    -----
    CSV file should have one of the following format.

    .. list-table:: CSV header format
       :widths: 20 20 20 20 20
       :header-rows: 1

       * - t
         - x
         - y
         - z
         - label
       * - required
         - required
         - required
         - optional
         - optional
    """

    objects = []
    with open(filename, "r") as csv_file:
        csvreader = csv.DictReader(csv_file, delimiter=",", quotechar="|")
        for i, row in enumerate(csvreader):
            data = {k: float(v) for k, v in row.items()}
            data["ID"] = i
            obj = btypes.PyTrackObject.from_dict(data)
            objects.append(obj)
    return objects
