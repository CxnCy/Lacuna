from lacuna.data.normalize import normalize_work


def test_normalize_work_keeps_required_fields():
    raw_work = {
        "id": "W123",
        "display_name": "Example Research Paper",
        "publication_year": 2015,
        "publication_date": "2015-06-01",
        "topics": [
            {
                "id": "T1",
                "display_name": "Machine Learning",
                "subfield": {"display_name": "Artificial Intelligence"},
                "field": {"display_name": "Computer Science"},
                "domain": {"display_name": "Physical Sciences"},
            }
        ],
        "authorships": [
            {
                "author": {
                    "id": "A1",
                }
            }
        ],
        "referenced_works": ["W100", "W101"],
        "cited_by_count": 42,
    }

    normalized = normalize_work(raw_work)

    assert normalized["work_id"] == "W123"
    assert normalized["publication_year"] == 2015
    assert normalized["topics"][0]["name"] == "Machine Learning"
    assert normalized["author_ids"] == ["A1"]
    assert normalized["referenced_work_ids"] == ["W100", "W101"]
    assert normalized["cited_by_count"] == 42


def test_normalize_work_handles_missing_optional_fields():
    raw_work = {
        "id": "W123",
        "display_name": "Minimal Paper",
        "publication_year": 2015,
        "publication_date": "2015-01-01",
    }

    normalized = normalize_work(raw_work)

    assert normalized["topics"] == []
    assert normalized["author_ids"] == []
    assert normalized["referenced_work_ids"] == []
    assert normalized["cited_by_count"] == 0