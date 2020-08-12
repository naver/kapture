# Copyright 2020-present NAVER Corp. Under BSD 3-clause license

from typing import Set, Tuple


class Matches(Set[Tuple[str, str]]):
    """
    Image matches
    """
    @staticmethod
    def lexical_order(image_path1: str, image_path2: str):
        """ returns the given pair in lexicographic order."""
        assert isinstance(image_path1, str)
        assert isinstance(image_path2, str)
        if image_path1 < image_path2:
            return image_path1, image_path2
        else:
            return image_path2, image_path1

    def add(self, image_path1: str, image_path2: str):
        """
        Inserts the image pair to matches. Does not ensure lexicographic order.
        Makes sure the entry guaranty that, or call normalize() afterward.

        :param image_path1:
        :param image_path2:
        :return:
        """
        assert isinstance(image_path1, str)
        assert isinstance(image_path2, str)
        # image_path1, image_path2 = self.lexical_order(image_path1, image_path2)
        super().add((image_path1, image_path2))

    def normalize(self):
        """ enforce lexicographic order on all matches. """
        temporary_set = Matches()
        for image_path1, image_path2 in self:
            image_path1, image_path2 = self.lexical_order(image_path1, image_path2)
            temporary_set.add(image_path1, image_path2)
        self.clear()
        self.update(temporary_set)

    def __repr__(self):
        if len(self) == 0:
            representation = 'no matches'
        else:
            representation = '[\n'
            representation += ',\n'.join(f'\t({image_path1} , {image_path2})' for (image_path1, image_path2) in self)
            representation += '\n]'
        return representation
