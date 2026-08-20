import sys

with open('frontend/src/App.test.tsx', 'r', encoding='utf-8') as f:
    lines = f.readlines()

with open('frontend/src/App.test.tsx', 'w', encoding='utf-8') as f:
    for line in lines:
        if "toHaveTextContent('Backend: ? Connected');" in line:
            f.write("      expect(screen.getByTestId('backend-status')).toHaveTextContent(/Connected/);\n")
        elif "toHaveTextContent('Backend: ? Offline');" in line:
            f.write("      expect(screen.getByTestId('backend-status')).toHaveTextContent(/Offline/);\n")
        else:
            f.write(line)
