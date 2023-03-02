import Avatar from "./avatar";
import Menu from "./menu";

function Header() {
  return (
    <div className="header_container">
      <div className="wrapper">
        <header className="header">
          <Avatar />
          <Menu />
        </header>
      </div>
    </div>
  );
}
export default Header;
