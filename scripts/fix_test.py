import sys

with open('frontend/src/App.test.tsx', 'r', encoding='utf-8') as f:
    lines = f.readlines()

with open('frontend/src/App.test.tsx', 'w', encoding='utf-8') as f:
    for line in lines:
        if "expect(screen.getByText(/Backend: ? Connected/)).toBeInTheDocument();" in line:
            f.write("      expect(screen.getByTestId('backend-status')).toHaveTextContent('Backend: ? Connected');\n")
        elif "expect(screen.getByText(/Backend: ? Offline/)).toBeInTheDocument();" in line:
            f.write("      expect(screen.getByTestId('backend-status')).toHaveTextContent('Backend: ? Offline');\n")
        else:
            f.write(line)
