#The Group class manages the groups of features as they are created.

class Groups:

    all_groups = [] 

    def __init__(self,leader):
        self.leader = leader
        self.members = [leader]
        Groups.all_groups.append(self)

    #Add a member to a group
    def add_member(self,feature):
        self.members.append(feature)

    #Remove a member from the group
    def remove_member(self,feature):
        self.members.remove(feature)

    #Returns a list of the members of a group
    def get_members(self):
        return self.members

    #Method that checks if a given feature is in this group
    def is_in(self,feature):
        return feature in self.members

    #A class Method that checks if a feature is in any group 
    @classmethod
    def is_grouped(cls,feature):
        for grp in Groups.all_groups:
            if(grp.is_in(feature)):
                return True
        return False

    #A class Method that returns the group object for a given feature
    @classmethod
    def get_group(cls,feature):
        for grp in Groups.all_groups:
            if(grp.is_in(feature)):
                return grp
        return False

    @classmethod
    def print_all_groups(cls):
        lst = []
        for grp in Groups.all_groups:
            lst.append(grp.get_members())

        return lst

