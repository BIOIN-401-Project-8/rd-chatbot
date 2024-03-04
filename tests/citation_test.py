from src.citation import get_sources


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
