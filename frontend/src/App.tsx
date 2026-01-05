import { useState } from 'react'
// import reactLogo from './assets/react.svg'
// import viteLogo from '/vite.svg'
import './App.css'
import PatientForm from './components/patientforms.tsx'
// =================================================


function App() {
  const [count, setCount] = useState(0)
  const [isLoading, setIsLoading] = useState(false);

  // Dummy run function for demonstration
  const run = () => {
    setIsLoading(true);
    // Simulate async operation
    setTimeout(() => setIsLoading(false), 1000);
  };

  return (
    <>
      <PatientForm isLoading={isLoading} onSubmit={run} />
    </>
  )
}

export default App
