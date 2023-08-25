if (show) {
    this._view = this.childTemplate.view(this.controller);
    this.section.appendChild(this._view.render());
}






/* Commit message: 
change to scope Delete bind: function (controllert) {
this.controller = controllert; Add bind: function (scopet) {
this.scope = scopet;
*/

// Action:
v = v.evaluate(this.view, this.controller);
// + v = v.evaluate(this.view, this.scope); 

v = v.evaluate(this.view, this.controller);
// + v = v.evaluate(this.view, this.scope); 

bind: function (controllert) {
    Base.prototype.bind.call(this, controllert);
// + bind: function (scopet) {
//     Base.prototype.bind.call(this, scopet); 