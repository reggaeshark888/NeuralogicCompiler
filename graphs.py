import schemdraw
import schemdraw.logic as logic

with schemdraw.Drawing() as d:
    # Draw OR gate
    or_gate = d.add(logic.Or().label('OR'))
    or_gate.input1.at((0, 2))
    or_gate.input2.at((0, 1))
    
    # Draw NAND gate
    nand_gate = d.add(logic.Nand().label('NAND').anchor('in1').at(or_gate.input1))
    nand_gate.input2.at((0, 0))
    
    # Draw AND gate
    and_gate = d.add(logic.And().label('AND').at(or_gate.output).right().label("Output", loc='right'))
    d.add(logic.Line().left().at(and_gate.input1).to(or_gate.output))
    d.add(logic.Line().left().at(and_gate.input2).to(nand_gate.output))
    
    # Inputs a and b
    d.add(logic.Line().left().at(nand_gate.input1).to((0, 2)).label('a', loc='left'))
    d.add(logic.Line().left().at(nand_gate.input2).to((0, 1)).label('b', loc='left'))
    d.add(logic.Line().left().at(or_gate.input1).to((0, 2)))
    d.add(logic.Line().left().at(or_gate.input2).to((0, 1)))

d.draw()
