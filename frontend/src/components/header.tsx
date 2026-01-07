export default function Header() {
  return (
    <header className="fixed top-0 z-50 w-full border-b border-zinc-200 bg-white">
      <div className="mx-auto flex h-14 items-center justify-between px-20">
        <div className="flex items-center">
          <img
            src="/logo.svg"
            alt="Strata Logo"
            className="h-[50px] w-[50px] object-contain"
          />
        </div>

        <nav className="hidden sm:block">
          <ul className="flex gap-8 text-bold">
            <li>
              <a className="text-sm text-zinc-800 hover:text-blue-600" href="/">
                Home
              </a>
            </li>
            <li>
              <a className="text-sm text-zinc-800 hover:text-blue-600" href="/about">
                About
              </a>
            </li>
            <li>
              <a className="text-sm text-zinc-800 hover:text-blue-600" href="/contact">
                Contact
              </a>
            </li>
          </ul>
        </nav>
      </div>
    </header>
  );
}
