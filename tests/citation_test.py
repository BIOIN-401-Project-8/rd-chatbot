from src.citation import expand_citations, format_citation, generate_full_pmid_citation, get_sources


class TestCitation:
    def test_get_sources(self):
        sources = get_sources(
            "Duchenne muscular dystrophy (DMD) is a type of muscular dystrophy that falls under the category of "
            "X-linked recessive diseases (SOURCE 3). It is also a subclass of muscular dystrophy (SOURCE 2) and has "
            "various manifestations, including muscle weakness (SOURCE 7), distrofia muscular congénita (SOURCE 6), "
            "dilated cardiomyopathy (SOURCE 9), and hypertrophic cardiomyopathy (SOURCE 10). DMD is also known as "
            "Duchenne muscular dystrophy, DMD, muscular dystrophy, Duchenne type, pseudo-hypertrophic progressive, "
            "Duchenne type, and congenital muscular dystrophy (SOURCES 11, 12, 15)."
        )
        assert sources == {2, 3, 6, 7, 9, 10, 11, 12, 15}

    def test_expand_citations(self):
        assert expand_citations("Sources 3, 8, and 9") == "Source 3, Source 8, Source 9"

    def test_generate_full_pmid_citation(self):
        assert (
            generate_full_pmid_citation("11561226")
            == "Jakobs, P. M., Hanson, E. L., Crispell, K. A., Toy, W., Keegan, H., Schilling, K., … Hershberger, R. E. (2001 , Sep). Novel lamin a/c mutations in two families with dilated cardiomyopathy and conduction system disease. Journal of cardiac failure. URL: https://pubmed.ncbi.nlm.nih.gov/11561226/, doi:10.1054/jcaf.2001.26339"
        )

    def test_format_citation(self):
        assert (
            format_citation("PMID:11561226")
            == "Jakobs, P. M., Hanson, E. L., Crispell, K. A., Toy, W., Keegan, H., Schilling, K., … Hershberger, R. E. (2001 , Sep). Novel lamin a/c mutations in two families with dilated cardiomyopathy and conduction system disease. Journal of cardiac failure. URL: https://pubmed.ncbi.nlm.nih.gov/11561226/, doi:10.1054/jcaf.2001.26339"
        )
