import json
import os 

AMENDMENTS = {'first': 'Congress shall make no law respecting an establishment of religion, or prohibiting the free exercise thereof; or abridging the freedom of speech, or of the press; or the right of the people peaceably to assemble, and to petition the Government for a redress of grievances.',
              'second':'A well regulated Militia, being necessary to the security of a free State, the right of the people to keep and bear Arms, shall not be infringed.A well regulated Militia, being necessary to the security of a free State, the right of the people to keep and bear Arms, shall not be infringed.',
              'third': 'No Soldier shall, in time of peace be quartered in any house, without the consent of the Owner, nor in time of war, but in a manner to be prescribed by law.',
              'fourth':'The right of the people to be secure in their persons, houses, papers, and effects, against unreasonable searches and seizures, shall not be violated, and no Warrants shall issue, but upon probable cause, supported by Oath or affirmation, and particularly describing the place to be searched, and the persons or things to be seized.', 
              'fifth': 'No person shall be held to answer for a capital, or otherwise infamous crime, unless on a presentment or indictment of a Grand Jury, except in cases arising in the land or naval forces, or in the Militia, when in actual service in time of War or public danger; nor shall any person be subject for the same offence to be twice put in jeopardy of life or limb; nor shall be compelled in any criminal case to be a witness against himself, nor be deprived of life, liberty, or property, without due process of law; nor shall private property be taken for public use, without just compensation.',
              'sixth':'In all criminal prosecutions, the accused shall enjoy the right to a speedy and public trial, by an impartial jury of the State and district wherein the crime shall have been committed, which district shall have been previously ascertained by law, and to be informed of the nature and cause of the accusation; to be confronted with the witnesses against him; to have compulsory process for obtaining witnesses in his favor, and to have the Assistance of Counsel for his defence.',
              'seventh': 'In Suits at common law, where the value in controversy shall exceed twenty dollars, the right of trial by jury shall be preserved, and no fact tried by a jury, shall be otherwise re-examined in any Court of the United States, than according to the rules of the common law.', 
              'eight': 'Excessive bail shall not be required, nor excessive fines imposed, nor cruel and unusual punishments inflicted.', 
              'nineth': 'The enumeration in the Constitution, of certain rights, shall not be construed to deny or disparage others retained by the people.',
              'tenth': 'The powers not delegated to the United States by the Constitution, nor prohibited by it to the States, are reserved to the States respectively, or to the people.',
              'eleventh': 'The Judicial power of the United States shall not be construed to extend to any suit in law or equity, commenced or prosecuted against one of the United States by Citizens of another State, or by Citizens or Subjects of any Foreign State.',
              'twelve': 'The Electors shall meet in their respective states and vote by ballot for President and Vice-President, one of whom, at least, shall not be an inhabitant of the same state with themselves; they shall name in their ballots the person voted for as President, and in distinct ballots the person voted for as Vice-President, and they shall make distinct lists of all persons voted for as President, and of all persons voted for as Vice-President, and of the number of votes for each, which lists they shall sign and certify, and transmit sealed to the seat of the government of the United States, directed to the President of the Senate; -- the President of the Senate shall, in the presence of the Senate and House of Representatives, open all the certificates and the votes shall then be counted; -- The person having the greatest number of votes for President, shall be the President, if such number be a majority of the whole number of Electors appointed; and if no person have such majority, then from the persons having the highest numbers not exceeding three on the list of those voted for as President, the House of Representatives shall choose immediately, by ballot, the President. But in choosing the President, the votes shall be taken by states, the representation from each state having one vote; a quorum for this purpose shall consist of a member or members from two-thirds of the states, and a majority of all the states shall be necessary to a choice. [And if the House of Representatives shall not choose a President whenever the right of choice shall devolve upon them, before the fourth day of March next following, then the Vice-President shall act as President, as in case of the death or other constitutional disability of the President. --]* The person having the greatest number of votes as Vice-President, shall be the Vice-President, if such number be a majority of the whole number of Electors appointed, and if no person have a majority, then from the two highest numbers on the list, the Senate shall choose the Vice-President; a quorum for the purpose shall consist of two-thirds of the whole number of Senators, and a majority of the whole number shall be necessary to a choice. But no person constitutionally ineligible to the office of President shall be eligible to that of Vice-President of the United States.',
              'thirteenth': 'Neither slavery nor involuntary servitude, except as a punishment for crime whereof the party shall have been duly convicted, shall exist within the United States, or any place subject to their jurisdiction. Congress shall have power to enforce this article by appropriate legislation. '}


class data():
    def __init__(self, directory: str, train_test: str) -> None:
        self.directory = directory
        self.train_test = train_test
        self.dir_list = None
        self.observations = None
        self.amendments_categories = None
        

    def dir_files(self):
        self.dir_list = [x for x in os.listdir(self.directory) if self.train_test in x] 
        return self.dir_list


    def org_data(self):
        self.observations = []
        for file in self.dir_list:
            f = open(file, 'r')
            file_contents = f.read().splitlines()
            for elem in file_contents:
                self.observations.extend(json.loads(elem))
        return self.observations


    def label(self):
        for i,sample in enumerate(self.observations):
            sample['label'] = 0
            if len(sample['matches']) != 0:
                sample['label'] = 1
            # print(i, sample, type(sample))
        return self.observations


    def check_amendments(self):
        self.amendments_categories = []
        for x in self.observations:
            if x['label'] == 1:
                if x['matches'][0][0] not in self.amendments_categories:
                    self.amendments_categories.append(x['matches'][0][0])
        return self.amendments_categories 


    def add_amendments(self):
        for sample in self.observations:
            if sample['label'] == 0:
                sample['amendment'] = 0
            if sample['label'] == 1:
                if 'fir' in sample['matches'][0][0]:
                    sample['amendment'] == 1
                if 'sec' in sample['matches'][0][0]:
                    sample['amendment'] == 2
                if 'thir' in sample['matches'][0][0]:
                    sample['amendment'] == 3
                if 'four' in sample['matches'][0][0]:
                    sample['amendment'] == 4
                if 'fift' in sample['matches'][0][0]:
                    sample['amendment'] == 5
                if 'sixt' in sample['matches'][0][0]:
                    sample['amendment'] == 6

        return self.observations


if __name__ == "__main__":
    path = os.getcwd()
    training = data(path, 'full')
    # print(training.directory, '\n', training.train_test)
    dir_list = training.dir_files()
    # print(dir_list)
    obs = training.org_data()
    print(len(obs), obs[:3])
    print(len(training.observations))
    obs2 = training.label()
    print(training.observations[:3])

    categoriaas = training.check_amendments()
    print(categoriaas)
    categorias = training.add_amendments()





