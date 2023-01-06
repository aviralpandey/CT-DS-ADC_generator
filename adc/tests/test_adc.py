import numpy as np
from adc import data_dir


def test_db():
    """Test `database_query`"""
    from ..database_query import (
        MosDB,
        nch_db_filename,
        nch_lvt_db_filename,
        pch_db_filename,
        pch_lvt_db_filename,
    )

    db = MosDB()
    for filename in [
        nch_db_filename,
        nch_lvt_db_filename,
        pch_db_filename,
        pch_lvt_db_filename,
    ]:
        data = np.load(
            data_dir / filename, allow_pickle=True
        )  # FIXME: does this do anything?
        db.build(data_dir / filename)


if __name__ == "__main__":
    test_db()
